import textwrap


def print_wrap(text: str, width: int = 90, subsequent_indent: str = "ꜛ"):
    """
    Wrap and print text to some max width.

    Useful for notebooks which do not wrap output text, making it very hard to read
    in both Colab and GitHub.
    """
    print(
        "\n".join(
            textwrap.fill(line, subsequent_indent=subsequent_indent, width=width)
            for line in text.split("\n")
        )
    )
