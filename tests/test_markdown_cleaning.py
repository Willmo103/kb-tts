from kb_tts.api import clean_markdown_text

def test_clean_markdown_text_none_or_empty():
    assert clean_markdown_text(None) is None
    assert clean_markdown_text("") == ""

def test_clean_markdown_text_bold_and_italics():
    # Bold
    assert clean_markdown_text("This is **bold** text.") == "This is bold text."
    assert clean_markdown_text("This is __bold__ text.") == "This is bold text."
    # Italics
    assert clean_markdown_text("This is *italic* text.") == "This is italic text."
    assert clean_markdown_text("This is _italic_ text.") == "This is italic text."
    # Combined
    assert clean_markdown_text("This is **bold** and *italic*.") == "This is bold and italic."

def test_clean_markdown_text_headers():
    assert clean_markdown_text("# Heading 1") == "Heading 1"
    assert clean_markdown_text("### Heading 3") == "Heading 3"
    assert clean_markdown_text("  ## Heading 2  ") == "Heading 2"

def test_clean_markdown_text_links_and_images():
    # Link
    assert clean_markdown_text("Check [this link](https://example.com) for more details.") == "Check this link for more details."
    # Image
    assert clean_markdown_text("Here is an image: ![Alt Text](https://example.com/img.png)") == "Here is an image: Alt Text"

def test_clean_markdown_text_inline_code():
    assert clean_markdown_text("Run `pip install piper-tts` to install.") == "Run pip install piper-tts to install."

def test_clean_markdown_text_code_blocks_keep():
    text = """Before code
```python
def hello():
    print("world")
```
After code"""
    expected = """Before code
def hello():
    print("world")
After code"""
    assert clean_markdown_text(text, strip_code_blocks=False) == expected

def test_clean_markdown_text_code_blocks_strip():
    text = """Before code
```python
def hello():
    print("world")
```
After code"""
    expected = """Before code

After code"""
    assert clean_markdown_text(text, strip_code_blocks=True) == expected

def test_clean_markdown_text_lists_and_quotes():
    # Bullet points
    text = """- Item 1
* Item 2
+ Item 3"""
    expected = """Item 1
Item 2
Item 3"""
    assert clean_markdown_text(text) == expected

    # Blockquotes
    text = """> This is a quote.
> And another line."""
    expected = """This is a quote.
And another line."""
    assert clean_markdown_text(text) == expected
