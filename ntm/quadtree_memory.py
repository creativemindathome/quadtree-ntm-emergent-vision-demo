"""Packed sparse quadtree memory with pointer-like logical addressing.

The serialized representation is deliberately minimal:

    uint32 logical_address | uint8 grayscale_value

Each record is exactly five bytes.  There are no stored child pointers and no
stored SPLIT bit.  A node is a split node precisely when all four addresses
``4*i+1 .. 4*i+4`` are present.  Missing child addresses encode an irregular
STOP decision.
"""

from dataclasses import dataclass
import struct
from typing import Dict, Iterable, Tuple

import numpy as np


_MAGIC = b"QTM1"
_HEADER = struct.Struct("<4sHHB3xI")
_RECORD = struct.Struct("<IB")


@dataclass(frozen=True)
class QuadtreeRead:
    """Decoded result of reading one logical quadtree address."""

    address: int
    physical_row: int
    value_u8: int
    depth: int
    x0: int
    y0: int
    size: int
    split: bool

    @property
    def value(self) -> float:
        return self.value_u8 / 255.0

    @property
    def child_addresses(self) -> Tuple[int, int, int, int]:
        return tuple(4 * self.address + offset for offset in (1, 2, 3, 4))


def full_tree_capacity(max_depth: int) -> int:
    return (4 ** (max_depth + 1) - 1) // 3


def parent_address(address: int) -> int:
    if address <= 0:
        raise ValueError("the root address has no parent")
    return (address - 1) // 4


def child_addresses(address: int) -> Tuple[int, int, int, int]:
    return tuple(4 * address + offset for offset in (1, 2, 3, 4))


def address_to_bounds(address: int, canvas_size: int) -> Tuple[int, int, int, int]:
    """Decode logical address into ``(depth, x0, y0, size)``."""
    if address < 0:
        raise ValueError("address must be non-negative")

    path = []
    cursor = address
    while cursor:
        path.append((cursor - 1) % 4)
        cursor = parent_address(cursor)

    x0 = 0
    y0 = 0
    size = canvas_size
    for quadrant in reversed(path):
        if size % 2:
            raise ValueError("canvas cannot be divided to the requested depth")
        size //= 2
        x0 += (quadrant & 1) * size
        y0 += ((quadrant >> 1) & 1) * size
    return len(path), x0, y0, size


class PackedQuadtreeMemory:
    """Sorted sparse records supporting address, node, and pixel reads."""

    def __init__(
            self,
            addresses: np.ndarray,
            values: np.ndarray,
            canvas_size: int = 128,
            valid_size: int = 100,
            max_depth: int = 7,
    ):
        self.addresses = np.asarray(addresses, dtype=np.uint32).reshape(-1).copy()
        self.values = np.asarray(values, dtype=np.uint8).reshape(-1).copy()
        self.canvas_size = int(canvas_size)
        self.valid_size = int(valid_size)
        self.max_depth = int(max_depth)
        self._validate()

    @classmethod
    def from_quadtree_sample(
            cls,
            sample: Dict,
            canvas_size: int = 128,
            valid_size: int = 100,
            max_depth: int = 7,
    ):
        """Quantize the mean column of an existing sparse training sample."""
        addresses = sample["heap_indices"].detach().cpu().numpy().astype(np.uint32)
        means = sample["memory"][:, 0].detach().cpu().numpy()
        values = np.rint(np.clip(means, 0.0, 1.0) * 255.0).astype(np.uint8)
        order = np.argsort(addresses)
        return cls(
            addresses[order],
            values[order],
            canvas_size=canvas_size,
            valid_size=valid_size,
            max_depth=max_depth,
        )

    @classmethod
    def from_bytes(cls, packed: bytes):
        """Load and validate the exact five-byte-record wire representation."""
        if len(packed) < _HEADER.size:
            raise ValueError("packed memory is shorter than its header")
        magic, canvas_size, valid_size, max_depth, count = _HEADER.unpack_from(packed, 0)
        if magic != _MAGIC:
            raise ValueError("invalid packed quadtree magic")
        expected_size = _HEADER.size + count * _RECORD.size
        if len(packed) != expected_size:
            raise ValueError("packed byte length does not match record count")

        addresses = np.empty(count, dtype=np.uint32)
        values = np.empty(count, dtype=np.uint8)
        offset = _HEADER.size
        for row in range(count):
            address, value = _RECORD.unpack_from(packed, offset)
            addresses[row] = address
            values[row] = value
            offset += _RECORD.size
        return cls(addresses, values, canvas_size, valid_size, max_depth)

    @property
    def record_count(self) -> int:
        return int(self.addresses.size)

    @property
    def record_size_bytes(self) -> int:
        return _RECORD.size

    @property
    def serialized_size_bytes(self) -> int:
        return _HEADER.size + self.record_count * _RECORD.size

    @property
    def capacity(self) -> int:
        return full_tree_capacity(self.max_depth)

    def _validate(self):
        if self.addresses.size != self.values.size:
            raise ValueError("addresses and values must have equal lengths")
        if not self.addresses.size or int(self.addresses[0]) != 0:
            raise ValueError("the root address 0 must be present")
        if self.canvas_size <= 0 or self.canvas_size & (self.canvas_size - 1):
            raise ValueError("canvas_size must be a power of two")
        if not 0 < self.valid_size <= self.canvas_size:
            raise ValueError("valid_size must be in (0, canvas_size]")
        if 2 ** self.max_depth != self.canvas_size:
            raise ValueError("max_depth must end in one-pixel canvas cells")

        if np.any(self.addresses[1:] <= self.addresses[:-1]):
            raise ValueError("logical addresses must be sorted and unique")
        if int(self.addresses[-1]) >= self.capacity:
            raise ValueError("logical address exceeds configured tree capacity")

        address_set = set(int(address) for address in self.addresses)
        for address in address_set:
            if address and parent_address(address) not in address_set:
                raise ValueError("every written node must have a written parent")
            present_children = sum(child in address_set for child in child_addresses(address))
            if present_children not in (0, 4):
                raise ValueError("a node must contain either zero or four children")

    def to_bytes(self) -> bytes:
        packed = bytearray(self.serialized_size_bytes)
        _HEADER.pack_into(
            packed,
            0,
            _MAGIC,
            self.canvas_size,
            self.valid_size,
            self.max_depth,
            self.record_count,
        )
        offset = _HEADER.size
        for address, value in zip(self.addresses, self.values):
            _RECORD.pack_into(packed, offset, int(address), int(value))
            offset += _RECORD.size
        return bytes(packed)

    def physical_row(self, address: int) -> int:
        """Translate a logical address to its compact row, or return -1."""
        row = int(np.searchsorted(self.addresses, np.uint32(address)))
        if row >= self.record_count or int(self.addresses[row]) != address:
            return -1
        return row

    def contains(self, address: int) -> bool:
        return self.physical_row(address) >= 0

    def is_split(self, address: int) -> bool:
        if not self.contains(address):
            raise KeyError("logical address {} is not written".format(address))
        # Validation guarantees that finding one child means finding all four.
        return self.contains(4 * address + 1)

    def read_node(self, address: int) -> QuadtreeRead:
        row = self.physical_row(address)
        if row < 0:
            raise KeyError("logical address {} is not written".format(address))
        depth, x0, y0, size = address_to_bounds(address, self.canvas_size)
        return QuadtreeRead(
            address=address,
            physical_row=row,
            value_u8=int(self.values[row]),
            depth=depth,
            x0=x0,
            y0=y0,
            size=size,
            split=self.is_split(address),
        )

    def leaf_addresses(self) -> Iterable[int]:
        for address in self.addresses:
            logical_address = int(address)
            if not self.is_split(logical_address):
                yield logical_address

    def read_pixel(self, x: int, y: int) -> QuadtreeRead:
        """Follow pointers from root until the leaf covering ``(x, y)``."""
        if not 0 <= x < self.valid_size or not 0 <= y < self.valid_size:
            raise IndexError("pixel is outside the valid image")

        address = 0
        while self.is_split(address):
            node = self.read_node(address)
            half = node.size // 2
            right = int(x >= node.x0 + half)
            bottom = int(y >= node.y0 + half)
            quadrant = right + 2 * bottom
            address = 4 * address + 1 + quadrant
            if not self.contains(address):
                raise RuntimeError("tree topology is incomplete during pixel read")
        return self.read_node(address)

    def render(self) -> np.ndarray:
        """Reconstruct the valid image using only packed memory reads."""
        canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        for address in self.leaf_addresses():
            node = self.read_node(address)
            canvas[node.y0:node.y0 + node.size, node.x0:node.x0 + node.size] = node.value_u8
        return canvas[:self.valid_size, :self.valid_size]
