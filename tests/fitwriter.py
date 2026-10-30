"""Minimal FIT encoder for tests (file_id + sport + record{timestamp, power}).

Implements just enough of the FIT protocol (14-byte header, definition and data
messages, CRC-16 as specified by the FIT SDK) for fitdecode to parse the file.
"""
import struct

_CRC = [0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
        0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400]


def crc16(data, crc=0):
    for b in data:
        t = _CRC[crc & 0xF]; crc = (crc >> 4) & 0x0FFF; crc ^= t ^ _CRC[b & 0xF]
        t = _CRC[crc & 0xF]; crc = (crc >> 4) & 0x0FFF; crc ^= t ^ _CRC[(b >> 4) & 0xF]
    return crc


def _definition(local, global_num, fields):
    out = struct.pack("<BBBHB", 0x40 | local, 0, 0, global_num, len(fields))
    for num, size, base in fields:
        out += struct.pack("<BBB", num, size, base)
    return out


def write_fit(path, power, t0=1_000_000_000, sport=2, step_s=1):
    body = _definition(0, 0, [(0, 1, 0x00), (4, 4, 0x86)])          # file_id: type, time_created
    body += struct.pack("<BBI", 0, 4, t0)
    body += _definition(1, 12, [(0, 1, 0x00)])                       # sport: sport
    body += struct.pack("<BB", 1, sport)
    body += _definition(2, 20, [(253, 4, 0x86), (7, 2, 0x84)])       # record: timestamp, power
    for i, p in enumerate(power):
        body += struct.pack("<BIH", 2, t0 + i * step_s, int(round(p)))
    header = struct.pack("<BBHI4s", 14, 0x20, 2132, len(body), b".FIT")
    header += struct.pack("<H", crc16(header))
    data = header + body
    with open(path, "wb") as f:
        f.write(data + struct.pack("<H", crc16(data)))
