import torch
import torch_ops
import unittest


class TestOps(unittest.TestCase):
    def test_square_plus_cpu(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.tensor([2.0, 6.0, 12.0, 20.0])
        result = torch_ops.square_plus(x)
        self.assertTrue(torch.allclose(result, expected))

    def test_square_plus_cuda(self):
        if not torch.cuda.is_available():
            return

        x = torch.tensor([1.0, 2.0, 3.0, 4.0]).cuda()
        expected = torch.tensor([2.0, 6.0, 12.0, 20.0]).cuda()
        result = torch_ops.square_plus(x)
        self.assertTrue(torch.allclose(result, expected))

    def test_modulo_cpu(self):
        x = torch.tensor([10.0, 15.0, 21.0, 7.0])
        expected = torch.tensor([0, 1, 1, 7], dtype=torch.int64)
        result = torch_ops.modulo(x, 10)
        self.assertTrue(torch.allclose(result, expected))

    def test_modulo_cuda(self):
        if not torch.cuda.is_available():
            return

        x = torch.tensor([10.0, 15.0, 21.0, 7.0]).cuda()
        expected = torch.tensor([0, 1, 1, 7], dtype=torch.int64).cuda()
        result = torch_ops.modulo(x, 10)
        self.assertTrue(torch.allclose(result, expected))


if __name__ == "__main__":
    unittest.main()