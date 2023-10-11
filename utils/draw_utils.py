'''
Utility functions for overlaying a caption on the meme template.
'''
from PIL import Image, ImageOps, ImageFont, ImageDraw
import matplotlib.pyplot as plt

def drawTextWithOutline(draw, font, text, x, y):
    draw.text((x-2, y-2), text,(0,0,0),font=font)
    draw.text((x+2, y-2), text,(0,0,0),font=font)
    draw.text((x+2, y+2), text,(0,0,0),font=font)
    draw.text((x-2, y+2), text,(0,0,0),font=font)
    draw.text((x, y), text, (255,255,255), font=font)
    return

def drawText(img, draw, font: ImageFont.FreeTypeFont, text, pos):
    text = text.upper()
    l, t, r, b = font.getbbox(text) # measure the size the text will take
    w = r - l
    h = b - t

    lineCount = 1
    if w > img.width:
        lineCount = int(round((w / img.width) + 1))

    lines = []
    if lineCount > 1:

        lastCut = 0
        isLast = False
        for i in range(0,lineCount):
            if lastCut == 0:
                cut = int((len(text) / lineCount) * i + 0.5)
            else:
                cut = lastCut

            if i < lineCount-1:
                nextCut = int((len(text) / lineCount) * (i+1) + 0.5)
            else:
                nextCut = len(text)
                isLast = True

            # make sure we don't cut words in half
            if not (nextCut == len(text) or text[nextCut] == " "):
                while text[nextCut] != " ":
                    nextCut += 1

            line = text[cut:nextCut].strip()

            # is line still fitting ?
            l, t, r, b = font.getbbox(line) # measure the size the text will take
            w = r - l
            h = b - t
            if not isLast and w > img.width:
                nextCut -= 1
                while text[nextCut] != " ":
                    nextCut -= 1

            lastCut = nextCut
            lines.append(text[cut:nextCut].strip())
    else:
        lines.append(text)

    lastY = -h
    if pos == "bottom":
        lastY = img.height - h * (lineCount+1) - 10

    for i in range(0, lineCount):
        l, t, r, b = font.getbbox(lines[i]) # measure the size the text will take
        w = r - l
        h = b - t
        x = img.width/2 - w/2
        y = lastY + h
        drawTextWithOutline(draw, font, lines[i], x, y)
        lastY = y

def draw_caption_and_display(img, response, return_img=False):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("data/impact.ttf", 20)
    if type(response) == str:
        words = response.split(" ")
        caption1 = " ".join(words[:len(words)//2])
        caption2 = " ".join(words[len(words)//2:])
        captions = [caption1, caption2]
    else:
        captions = response['choices'][0]['text'].split("<sep>")
    if len(captions) == 2:
        drawText(img, draw, font, captions[1], "bottom")
        drawText(img, draw, font, captions[0], "top")
    else:
        drawText(img, draw, font, captions[0], "top")
    if return_img:
        return img
    plt.imshow(img)
    plt.axis("off")
    plt.show()
