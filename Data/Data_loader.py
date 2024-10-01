from PyPDF2 import PdfReader

reader = PdfReader("Data/The Subtle Art.pdf")
page = reader.pages[0]
print(page.extract_text())