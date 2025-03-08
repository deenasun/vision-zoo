import h5py
import matplotlib.pyplot as plt

filepath = '/honey/nmep/medium-imagenet-96.hdf5'

# Open the HDF5 file
with h5py.File(filepath, 'r') as hf:
    # Assuming 'images' is a key containing your image data
    images = hf['images-test']
    
    # Load and display the first image
    first_image = images[0]
    
    # Display the image
    first_image = first_image.T
    plt.imshow(first_image)
    plt.show()
