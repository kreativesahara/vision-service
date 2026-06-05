import re

text = "MITSUBISHI KCG-26OP"
clean = re.sub(r'[^A-Z0-9]', '', text)
print(f"clean: {clean}")
# allow O, I, S, Z as digits
match = re.search(r'(K[A-Z]{2})([\dOISZ]{3}[A-Z]?)', clean)
if match:
    prefix = match.group(1)
    suffix = match.group(2)
    # replace common OCR errors in suffix
    suffix = suffix.translate(str.maketrans('OISZ', '0152'))
    print(f"prefix: {prefix}, suffix: {suffix}")
else:
    print("No match")
