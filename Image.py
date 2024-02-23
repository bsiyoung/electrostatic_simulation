import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from matplotlib.colors import Normalize, LinearSegmentedColormap
from PIL import Image


class ImageModify:
    def __init__(self, ticks=None, title='Title', colors=[(0,0,1), (1,1,1), (1,0,0)]):
        self.title = title
        self.ticks = ticks
        self.colors = colors

    def run(self):
        # Read image file with PIL
        img_pil = Image.open('result.png')

        # Get image size
        width, height = img_pil.size

        # Convert PIL image to numpy array for matplotlib
        img = mpimg.pil_to_array(img_pil)

        # Create Normalizer instance for custom color map
        norm = Normalize(vmin=-3, vmax=3)

        # Create custom color map
        colors = self.colors # Create a list with the desired colors
        cmap = LinearSegmentedColormap.from_list("mycmap", colors) # Create a color map using a list

        # Create subplot
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=160)

        # Display image with custom colormap
        cax = ax.imshow(img, cmap=cmap, norm=norm)

        fontsize = min(width, height) * 0.02 # Adjust the scaling factor (0.02) as needed

        # Create color bar and set label
        cbar = fig.colorbar(cax, ticks=[-3, 0, 3], ax=ax, orientation='vertical', shrink=0.6, aspect=10)
        cbar.ax.set_yticklabels(['-3v', '0v', '3v'], fontsize=fontsize*0.5)
        cbar.set_label('Color Bar Label', labelpad=10)

        # Set axis labels and title
        ax.set_title(self.title, fontsize=fontsize)
        ax.set_xlabel('X', fontsize=fontsize)
        ax.set_ylabel('Y', fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize*0.5) # Set font size for x-axis ticks
        ax.tick_params(axis='y', labelsize=fontsize*0.5) # Set font size for y-axis ticks

        # Add grid
        ax.grid(linewidth=fontsize*0.03)

        # Save and display the figure
        plt.savefig('output.png', bbox_inches='tight', pad_inches=0)
        os.startfile('output.png')


if __name__ == '__main__':
    img = ImageModify
    img.run()