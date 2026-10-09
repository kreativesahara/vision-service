"""After a deploy: analyse a made-up plate on the live service, and fail unless the plate comes back.

/health answers even when the services failed to load, so this is the check that analysis really works end to end
(Google Vision, Gemini, the plate logic). Each run makes a few Google calls, costing well under a cent.

Usage: python smoke_test.py https://vision.kemotives.co.ke
"""
import io
import json
import sys
import time
import urllib.request
import uuid

from PIL import Image, ImageDraw, ImageFont

PLATE = 'KAA 123A'  # made-up registration


def plate_image() -> bytes:
    image = Image.new('RGB', (900, 300), (240, 200, 40))
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 150)
    except OSError:
        font = ImageFont.load_default(size=150)
    ImageDraw.Draw(image).text((40, 60), PLATE, fill=(0, 0, 0), font=font)
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG')
    return buffer.getvalue()


def main(base_url: str) -> None:
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="images"; filename="plate.jpg"\r\n'
        'Content-Type: image/jpeg\r\n\r\n'
    ).encode() + plate_image() + f'\r\n--{boundary}--\r\n'.encode()
    request = urllib.request.Request(
        base_url.rstrip('/') + '/analyse',
        data=body,
        method='POST',
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
    )

    start = time.time()
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.load(response)
    plate = (data.get('plate') or {}).get('full_plate')
    print(f"/analyse took {time.time() - start:.1f}s (processing_time_ms={data.get('processing_time_ms')}), plate={plate!r}")

    if plate != PLATE:
        sys.exit(f'Smoke test failed: expected plate {PLATE!r}, got {plate!r}')


if __name__ == '__main__':
    main(sys.argv[1])
