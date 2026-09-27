import torch


def rmsnorm(x, gamma, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * gamma


def main():
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    gamma = torch.tensor([1.0, 2.0])
    eps = 1e-6

    output = rmsnorm(x, gamma, eps)

    # The first row has mean square 12.5; the zero row stays zero.
    expected = torch.tensor(
        [[3.0 / (12.5 + eps) ** 0.5, 8.0 / (12.5 + eps) ** 0.5],
         [0.0, 0.0]]
    )
    assert output.shape == x.shape, f"Unexpected shape: {output.shape}"
    torch.testing.assert_close(output, expected)

    print(f"Output:\n{output}")
    print(f"Output shape: {output.shape}")
    print("All checks passed!")


if __name__ == "__main__":
    main()
