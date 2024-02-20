# Importing library
import qrcode

# Data to be encoded
# data = 'coep'
data = 'https://forms.gle/k2BsXqrPA62g4hA4A'

# Encoding data using make() function
img = qrcode.make(data)

# Saving as an image file
img.save('MyQRCode1.png')

#print("hellooooo")
