import click
import os

from starter_library.templating import create_new_notebook_from_template

@click.command()
@click.option('--title', default="Your great analysis",help='Enter the name of your analysis')
def main(title):
    current_folder = os.path.abspath(os.path.dirname(__file__)) # Starter folder
    create_new_notebook_from_template(os.path.join(current_folder,"assets/template_notebook.ipynb"),title)

if __name__ == '__main__':
    main()