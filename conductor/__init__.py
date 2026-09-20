"""Conductor core: one monotonic clock, discovery, the UDP hub and the wire codec.

Written 2026-09-19 for Feel the Music. Standard library only; the optional
``zeroconf`` package is imported lazily by ``conductor.discovery.advertise``.
"""
PROTOCOL_VERSION = 1
