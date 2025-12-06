# runner_small_tensor_test.py
from image.bitmap import Bitmap
from scorch.scorch_tensor import ScorchTensor

def runner_small_tensor_test():

    bitmap = Bitmap.with_local_image(subdirectory="images", name="6_2_test.png")
    print("bitmap dims = (" + str(bitmap.width) + ", " + str(bitmap.height) + ")")

    st = ScorchTensor.from_bitmap(bitmap, "my_tiny_tensor", role="demo", grayscale=False)

    print("original shape...")
    print(st.shape)
    print("...")

    print("printing original...")
    print(st)
    print("done printing original...")

    flat = st.flatten()

    print("printing flat...")
    print(flat)
    print("done printing flat...")

