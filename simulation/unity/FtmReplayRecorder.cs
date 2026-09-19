// Written 2026-09-19. Original adapter; contains no vendor assets or runtime code.
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

/// <summary>
/// Records actual scene probes, not an IK solver or collision geometry model.
/// The replay driver must sample every physics/replay step, including the initial
/// and final state. The model manifest must identify/hash the actual loaded rig,
/// meshes/colliders, calibration, joint limits, environment and physics settings.
/// Hashing supplied bytes alone cannot prove those assets were loaded in Unity.
/// Call AbortReplay on caught driver/probe exceptions; logged Unity exceptions
/// and disabling this component also invalidate an unfinished recording.
/// </summary>
public sealed class FtmReplayRecorder : MonoBehaviour
{
    public const int MaxSamples = 100000;
    public const int MaxReportBytes = 16 * 1024 * 1024;
    private enum State { Ready, Recording, Finished, Failed }
    private State state = State.Ready;
    private readonly List<Sample> samples = new List<Sample>();
    private Report report;

    [Serializable]
    private sealed class Sample
    {
        public long time_ns;
        public double clearance_m;
        public double head_error_deg;
        public bool collision;
        public bool joint_limit_violation;
        public bool tracking_valid;
    }

    [Serializable]
    private sealed class Report
    {
        public int schema_version = 1;
        public string scene_id;
        public string model_sha256;
        public string trajectory_sha256;
        public string simulator_version;
        public string mode;
        public bool completed;
        public int sample_count;
        public long duration_ns;
        public double min_clearance_m;
        public double max_head_error_deg;
        public int collision_count;
        public int joint_limit_violations;
        public int tracking_lost_count;
        public Sample[] samples;
    }

    private void OnEnable() { Application.logMessageReceived += OnLog; }
    private void OnDisable()
    {
        Application.logMessageReceived -= OnLog;
        AbortReplay();
    }

    private void OnLog(string message, string stackTrace, LogType type)
    {
        if (type == LogType.Exception || type == LogType.Assert || type == LogType.Error)
            AbortReplay();
    }

    // Explicitly discard previous recording state before starting another replay.
    public void ResetReplay()
    {
        state = State.Ready;
        samples.Clear();
        report = null;
    }

    public void AbortReplay()
    {
        if (state == State.Recording) state = State.Failed;
    }

    public void BeginReplay(string sceneId, byte[] modelManifestBytes,
                            byte[] trajectoryBytes, string mode)
    {
        try
        {
            if (!isActiveAndEnabled || state != State.Ready)
                throw new InvalidOperationException("Enable and reset the recorder before beginning.");
            if (string.IsNullOrWhiteSpace(sceneId))
                throw new ArgumentException("A nonempty scene identifier is required.");
            if (mode != "head-follow" && mode != "dance")
                throw new ArgumentException("Mode must be head-follow or dance.");
            if (modelManifestBytes == null || modelManifestBytes.Length == 0 ||
                trajectoryBytes == null || trajectoryBytes.Length == 0)
                throw new ArgumentException("Exact nonempty manifest and trajectory bytes are required.");
            report = new Report {
                scene_id = sceneId,
                model_sha256 = Hash(modelManifestBytes),
                trajectory_sha256 = Hash(trajectoryBytes),
                simulator_version = Application.unityVersion,
                mode = mode
            };
            state = State.Recording;
        }
        catch { state = State.Failed; throw; }
    }

    public void RecordSample(long time_ns, double clearance_m, double head_error_deg,
                             bool collision, bool joint_limit_violation, bool tracking_valid)
    {
        try
        {
            RequireRecording();
            if (samples.Count >= MaxSamples)
                throw new InvalidOperationException("Sample limit reached; recording is incomplete.");
            if (time_ns < 0 || (samples.Count > 0 && time_ns <= samples[samples.Count - 1].time_ns))
                throw new ArgumentException("Sample time must be nonnegative and strictly increasing.");
            if (!Finite(clearance_m) || !Finite(head_error_deg) || head_error_deg < 0)
                throw new ArgumentException("Probes must be finite; head error must be nonnegative.");
            samples.Add(new Sample {
                time_ns = time_ns, clearance_m = clearance_m, head_error_deg = head_error_deg,
                collision = collision, joint_limit_violation = joint_limit_violation,
                tracking_valid = tracking_valid
            });
        }
        catch { state = State.Failed; throw; }
    }

    // Expected endpoints come from the trajectory, not observed samples.
    // Call only after full playback. Existing evidence is never overwritten.
    public void FinishReplay(string path, long expectedStartTimeNs, long expectedFinalTimeNs)
    {
        string temporaryPath = null;
        try
        {
            RequireRecording();
            if (samples.Count < 2 || expectedStartTimeNs < 0 ||
                expectedFinalTimeNs <= expectedStartTimeNs ||
                samples[0].time_ns != expectedStartTimeNs ||
                samples[samples.Count - 1].time_ns != expectedFinalTimeNs)
                throw new InvalidOperationException("The complete replay needs at least two samples and both expected endpoint times.");
            report.samples = samples.ToArray();
            report.sample_count = samples.Count;
            report.duration_ns = samples[samples.Count - 1].time_ns - samples[0].time_ns;
            report.min_clearance_m = double.PositiveInfinity;
            report.max_head_error_deg = 0;
            foreach (Sample sample in samples)
            {
                report.min_clearance_m = Math.Min(report.min_clearance_m, sample.clearance_m);
                report.max_head_error_deg = Math.Max(report.max_head_error_deg, sample.head_error_deg);
                if (sample.collision) report.collision_count++;
                if (sample.joint_limit_violation) report.joint_limit_violations++;
                if (!sample.tracking_valid) report.tracking_lost_count++;
            }
            report.completed = true; // Completion is not a safety verdict.
            byte[] serialized = new UTF8Encoding(false).GetBytes(JsonUtility.ToJson(report, true));
            if (serialized.Length > MaxReportBytes)
                throw new InvalidOperationException("Serialized report exceeds the 16 MiB validator limit.");
            string outputPath = Path.GetFullPath(path);
            if (File.Exists(outputPath))
                throw new IOException("Choose a new report filename; existing evidence is preserved.");
            temporaryPath = outputPath + "." + Guid.NewGuid().ToString("N") + ".tmp";
            File.WriteAllBytes(temporaryPath, serialized);
            File.Move(temporaryPath, outputPath);
            temporaryPath = null;
            state = State.Finished;
        }
        catch
        {
            state = State.Failed;
            if (report != null) report.completed = false;
            throw;
        }
        finally
        {
            if (temporaryPath != null && File.Exists(temporaryPath))
                File.Delete(temporaryPath);
        }
    }

    private void RequireRecording()
    {
        if (!isActiveAndEnabled || state != State.Recording)
            throw new InvalidOperationException("No active, valid replay recording.");
    }

    private static bool Finite(double value)
    {
        return !double.IsNaN(value) && !double.IsInfinity(value);
    }

    private static string Hash(byte[] bytes)
    {
        using (SHA256 sha = SHA256.Create())
        {
            byte[] digest = sha.ComputeHash(bytes);
            StringBuilder hex = new StringBuilder(64);
            foreach (byte value in digest) hex.Append(value.ToString("x2"));
            return hex.ToString();
        }
    }
}
