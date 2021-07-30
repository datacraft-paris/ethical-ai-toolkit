import getpass
import datetime
import os
from jinja2 import Template


def easy_template(template_path,destination_path,params = None):
    if not os.path.exists(destination_path):
        if params is None: params = {}
        template = Template(open(template_path,"r").read())
        new_file = template.render(params)
        with open(destination_path,"w") as file:
            file.write(new_file)
        print(f"... Created file {destination_path} from {template_path}")
    else:
        print(f"... Skipped creation because {destination_path} already exists")




def create_new_notebook_from_template(template_path,title,params = None):
    creation_date = datetime.datetime.today().strftime("%Y%m%d")
    author = getpass.getuser()
    filepath = f"{creation_date} - {author} - {title}.ipynb"

    params = {"creation_date":creation_date,"author":author,"title":title}
    easy_template(template_path,filepath,params)


