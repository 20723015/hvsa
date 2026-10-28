"""Read-only observations; no power, frequency, affinity or priority changes."""
from __future__ import annotations
import datetime
import os
import time


def snapshot():
    import psutil
    battery=psutil.sensors_battery();frequency=psutil.cpu_freq()
    process=psutil.Process(os.getpid());cpu=process.cpu_times()
    others=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            if p.pid!=os.getpid() and (p.info['name'] or '').lower() in ['python.exe','python']:
                if any('experiments.point1_compare.' in x for x in (p.info['cmdline'] or [])):
                    others.append(p.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return dict(wall_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        monotonic_seconds=time.perf_counter(),
        battery=dict(percent=battery.percent,power_plugged=battery.power_plugged) if battery else None,
        cpu_frequency=frequency._asdict() if frequency else None,
        system_cpu_times=psutil.cpu_times()._asdict(),
        process_cpu_times=dict(user=cpu.user,system=cpu.system),
        process_rss_bytes=process.memory_info().rss,other_protocol_workers=others)


def summarize(entries):
    frames=[e[k] for e in entries for k in ['environment_before','environment_after']]
    power=[f['battery']['power_plugged'] if f['battery'] is not None else None for f in frames]
    freq=[f['cpu_frequency']['current'] for f in frames if f['cpu_frequency'] is not None]
    return dict(snapshots=len(frames),all_observed_power_plugged=bool(power) and all(x is True for x in power),
        observed_power_transitions=sum(a!=b for a,b in zip(power,power[1:])),
        snapshots_with_other_protocol_workers=sum(bool(f['other_protocol_workers']) for f in frames),
        reported_frequency_min_mhz=min(freq) if freq else None,reported_frequency_max_mhz=max(freq) if freq else None,
        temperature_and_frequency_locked=False,
        interpretation='Read-only observations, not complete hardware-condition control.')
