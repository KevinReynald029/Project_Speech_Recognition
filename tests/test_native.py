import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phoneme_scoring.native import read_matrix_archive


def encoded_matrix(key, matrix):
    return key.encode()+b" \0BFM \4"+struct.pack("<i",len(matrix))+b"\4"+struct.pack("<i",matrix.shape[1])+matrix.astype("<f4").tobytes()


class NativeArchiveTests(unittest.TestCase):
    def test_streams_multiple_real_kaldi_matrix_objects(self):
        a=np.array([[1.,2.,3.],[4.,5.,6.]],dtype=float)
        b=np.array([[-.1,-.2]],dtype=float)
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/"posterior.ark"
            file.write_bytes(encoded_matrix("000010011",a)+encoded_matrix("000010035",b))
            values=list(read_matrix_archive(file))
            self.assertEqual([key for key,_ in values],["000010011","000010035"])
            np.testing.assert_allclose(values[0][1],a)
            np.testing.assert_allclose(values[1][1],b)

    def test_rejects_truncated_or_text_archive(self):
        a=np.ones((2,3))
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/"posterior.ark"
            for raw in (encoded_matrix("a",a)[:-3],b"a [ 1 2 3 ]"):
                file.write_bytes(raw)
                with self.assertRaises(ValueError):
                    list(read_matrix_archive(file))
