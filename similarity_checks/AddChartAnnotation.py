from PIL import Image, ImageDraw, ImageFont

def add_text_to_png(image_path, text, output_path, font_path='/mnt/c/Windows/Fonts/arial.ttf', font_size=50, text_color=(255, 0, 0), position=(100, 20)):
    """
    Opens a PNG image, adds text annotation, and saves the result.

    Args:
        image_path (str): Path to the input PNG image.
        text (str): Text to add to the image.
        output_path (str): Path to save the output image.
        font_path (str, optional): Path to the font file. Defaults to 'arial.ttf'.
        font_size (int, optional): Font size. Defaults to 30.
        text_color (tuple, optional): Text color in RGB format. Defaults to (0, 0, 0) (black).
        position (tuple, optional): Position of the text (x, y). Defaults to (10, 10).
    """
    try:
        image = Image.open(image_path).convert("RGBA")
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return
    except Exception as e:
         print(f"An error occurred while opening the image: {e}")
         return

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"Error: Font file not found at '{font_path}'. Using default font.")
        font = ImageFont.load_default()
    except Exception as e:
        print(f"An error occurred while loading the font: {e}")
        return
    
    draw.text(position, text, fill=text_color, font=font)

    try:
        image.save(output_path, "PNG")
        print(f"Image with text saved to '{output_path}'")
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")

if __name__ == "__main__":
    datasets = ['dgemm', 'kmeans']
    algorithms = ['eseman_kdt', 'eseman_kdt_md', 'agglomerative_clustering']
    algorithms_names = {'eseman_kdt':'ESEMAN with Midpoint', 'eseman_kdt_md':'ESEMAN with Max Distance', 'agglomerative_clustering':'Agglomerative Clustering'}
    did = {'dgemm':'ecc21d0a-112a-4b52-8cdd-6aca80adde93','kmeans':'faf17535-2f66-4621-995f-49c7dbd84e8b'}
    bins = [1, 2, 4, 8, 16, 32, 1024]
    for algorithm in algorithms:
        for dataset in datasets:
            for bin in bins:
                file_name = './' + algorithm + '/' + dataset + '/' + did[dataset] + '_gantt_bin' + str(bin) + '.png'
                text_to_add = "Bin: " + str(bin) + "\n" + "Algorithm: " + algorithms_names[algorithm] + "\n" + "Dataset: " + dataset
                output_path = "./May19DataForCommittee/" + algorithm + "/" + dataset + '_gantt_bin' + str(bin) + '.png'
                add_text_to_png(file_name, text_to_add, output_path)
        #         break
        #     break
        # break
