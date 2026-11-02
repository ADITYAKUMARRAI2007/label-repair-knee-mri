"""Serialize accelerator durations without the installed SDK's day truncation.

Never reconstruct missing days from an old JSON receipt. Read typed timedelta
fields from a fresh response before budgeting a new launch.
"""
from datetime import timedelta

def duration_string(value):
    if not isinstance(value, timedelta):
        raise TypeError('Accelerator quota duration must be a typed timedelta')
    if value < timedelta(0):
        raise ValueError('Negative accelerator quota duration')
    seconds = value.days * 86400 + value.seconds
    fraction = f'.{value.microseconds:06d}'.rstrip('0') if value.microseconds else ''
    return f'{seconds}{fraction}s'

def quota_snapshot(response):
    """Preserve normal fields; replace every accelerator duration from its object."""
    result = response.to_dict()
    fields = [('timeUsed','time_used'), ('timeReserved','time_reserved'),
              ('totalTimeAllowed','total_time_allowed'),
              ('minimumTimeAllowed','minimum_time_allowed')]
    for key, attr in [('gpuQuota','gpu_quota'), ('tpuQuota','tpu_quota')]:
        quota = getattr(response, attr)
        if quota is None:
            continue
        for name, field in fields:
            value = getattr(quota, field)
            if value is not None:
                result.setdefault(key, {})[name] = duration_string(value)
    return result
