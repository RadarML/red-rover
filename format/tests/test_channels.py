"""Test cases for channels."""

import os
import shutil
import unittest

import numpy as np

from roverd import channels


class RawTestCase(unittest.TestCase):
    """RawChannel and base class."""

    PATH = '/tmp/rawchanneltest'

    def setUp(self):
        os.makedirs(self.PATH, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.PATH)

    def test_read(self):
        for dts in ['f8', 'u4', 'f4', 'i2', 'u1']:
            dt = np.dtype(dts)
            ch = channels.RawChannel(
                path=os.path.join(self.PATH, "test"),
                dtype=dts, shape=[10, 8, 6])

            data = (
                1000 * np.random.uniform(size=(12, 10, 8, 6))
            ).astype(dt)
            ch.write(data)

            self.assertEqual(ch.size, 10 * 8 * 6 * dt.itemsize)
            self.assertEqual(ch.filesize, data.size * dt.itemsize)

            self.assertTrue(np.all(ch.read() == data))
            self.assertTrue(np.all(ch.read(samples=2) == data[:2]))
            self.assertTrue(np.all(ch.read(start=5) == data[5:]))
            self.assertTrue(np.all(ch.read(start=8, samples=1) == data[8:9]))
            self.assertTrue(np.all(ch.memmap() == data))

    def test_write(self):
        ch = channels.RawChannel(
            path=os.path.join(self.PATH, "test"), dtype='f4', shape=[3, 5])
        data = np.random.uniform(size=(100, 3, 5)).astype(np.float32)

        ch.consume(data)
        self.assertTrue(np.all(ch.read() == data))
        ch.consume(x for x in data)
        self.assertTrue(np.all(ch.read() == data))

        ch.write(data, mode='ab')
        self.assertTrue(np.all(ch.read(start=100, samples=100) == data))
        self.assertTrue(ch.filesize, (data.size * 2) * 4)

        with self.assertRaises(ValueError):
            ch.write(data.astype(np.float64))
        with self.assertRaises(ValueError):
            ch.write(data[..., :4])

    def test_stream(self):
        ch = channels.RawChannel(
            path=os.path.join(self.PATH, "test"), dtype='f4', shape=[10])
        data = np.random.uniform(size=(100, 10)).astype(np.float32)
        ch.write(data)

        for actual, read in zip(data, ch.stream()):
            self.assertTrue(np.all(actual == read))

        for actual, read in zip(data, ch.stream_prefetch()):
            self.assertTrue(np.all(actual == read))

        for actual, read in zip(data, ch.stream(transform=lambda x: x * 2)):
            self.assertTrue(np.allclose(actual * 2, read))

        for actual, read in zip(data.reshape(20, 5, 10), ch.stream(batch=5)):
            self.assertTrue(np.all(actual == read))


class LzmafTestCase(unittest.TestCase):
    """LzmaFrameChannel."""

    PATH = '/tmp/lzmafchanneltest'

    def setUp(self):
        os.makedirs(self.PATH, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.PATH)

    def test_read(self):
        for dts in ['f8', 'u4', 'f4', 'i2', 'u1']:
            dt = np.dtype(dts)
            ch = channels.LzmaFrameChannel(
                path=os.path.join(self.PATH, "test"),
                dtype=dts, shape=[10, 8, 6])

            data = (
                1000 * np.random.uniform(size=(12, 10, 8, 6))
            ).astype(dt)
            ch.write(data)

            self.assertTrue(np.all(ch.read() == data))
            self.assertTrue(np.all(ch.read(samples=2) == data[:2]))
            self.assertTrue(np.all(ch.read(start=5) == data[5:]))
            self.assertTrue(np.all(ch.read(start=8, samples=1) == data[8:9]))

    def test_write(self):
        ch = channels.LzmaFrameChannel(
            path=os.path.join(self.PATH, "test"), dtype='f4', shape=[3, 5])
        data = np.random.uniform(size=(100, 3, 5)).astype(np.float32)

        ch.consume(data)
        self.assertTrue(np.all(ch.read() == data))
        ch.consume(x for x in data)
        self.assertTrue(np.all(ch.read() == data))

        with self.assertRaises(ValueError):
            ch.write(data.astype(np.float64))
        with self.assertRaises(ValueError):
            ch.write(data[..., :4])

    def test_stream(self):
        ch = channels.LzmaFrameChannel(
            path=os.path.join(self.PATH, "test"), dtype='f4', shape=[10])
        data = np.random.uniform(size=(100, 10)).astype(np.float32)
        ch.write(data)

        for actual, read in zip(data, ch.stream()):
            self.assertTrue(np.all(actual == read))

        for actual, read in zip(data, ch.stream_prefetch()):
            self.assertTrue(np.all(actual == read))

        for actual, read in zip(data, ch.stream(transform=lambda x: x * 2)):
            self.assertTrue(np.allclose(actual * 2, read))

        for actual, read in zip(data.reshape(20, 5, 10), ch.stream(batch=5)):
            self.assertTrue(np.all(actual == read))



if __name__ == '__main__':
    unittest.main()
