from __future__ import annotations

from collections import Counter
import struct


MASK = 0xFFFFFFFF
TOP = 0x80000000


class _BitWriter:
    def __init__(self) -> None: self.bits: list[int] = []
    def write(self, value: int) -> None: self.bits.append(value & 1)
    def finish(self) -> bytes:
        out = bytearray((len(self.bits) + 7) // 8)
        for index, bit in enumerate(self.bits): out[index // 8] |= bit << (7 - index % 8)
        return bytes(out)


class _BitReader:
    def __init__(self, data: bytes) -> None: self.data, self.index = data, 0
    def read(self) -> int:
        bit = (self.data[self.index // 8] >> (7 - self.index % 8)) & 1 if self.index // 8 < len(self.data) else 0
        self.index += 1
        return bit


def encode(values: list[int]) -> bytes:
    """Static arithmetic coder. The per-stream symbol table is part of the payload."""
    if not values: return struct.pack("<HI", 0, 0)
    counts = Counter(values)
    symbols = sorted(counts)
    frequencies = [counts[symbol] for symbol in symbols]
    index = {symbol: i for i, symbol in enumerate(symbols)}
    cumulative = [0]
    for frequency in frequencies: cumulative.append(cumulative[-1] + frequency)
    total = cumulative[-1]
    low, high, writer = 0, MASK, _BitWriter()
    for value in values:
        slot = index[value]
        distance = high - low + 1
        high = low + (distance * cumulative[slot + 1] // total) - 1
        low = low + (distance * cumulative[slot] // total)
        while (low & TOP) == (high & TOP):
            writer.write(high >> 31); low = (low << 1) & MASK; high = ((high << 1) & MASK) | 1
    for _ in range(32): writer.write(low >> 31); low = (low << 1) & MASK
    header = struct.pack("<HI", len(symbols), len(values))
    table = b"".join(struct.pack("<hI", symbol, frequency) for symbol, frequency in zip(symbols, frequencies))
    return header + table + writer.finish()


def decode(blob: bytes) -> list[int]:
    count, item_count = struct.unpack_from("<HI", blob)
    if item_count == 0: return []
    offset, symbols, frequencies = 6, [], []
    for _ in range(count):
        symbol, frequency = struct.unpack_from("<hI", blob, offset); offset += 6
        symbols.append(symbol); frequencies.append(frequency)
    cumulative = [0]
    for frequency in frequencies: cumulative.append(cumulative[-1] + frequency)
    total, reader = cumulative[-1], _BitReader(blob[offset:])
    low, high, code = 0, MASK, 0
    for _ in range(32): code = ((code << 1) & MASK) | reader.read()
    output: list[int] = []
    for _ in range(item_count):
        distance = high - low + 1
        target = ((code - low + 1) * total - 1) // distance
        slot = next(i for i in range(len(symbols)) if cumulative[i] <= target < cumulative[i + 1])
        output.append(symbols[slot])
        high = low + (distance * cumulative[slot + 1] // total) - 1
        low = low + (distance * cumulative[slot] // total)
        while (low & TOP) == (high & TOP):
            low = (low << 1) & MASK; high = ((high << 1) & MASK) | 1; code = ((code << 1) & MASK) | reader.read()
    return output
