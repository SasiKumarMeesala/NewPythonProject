from setuptools import setup, find_packages
from typing import List



def get_requirements(file_path: str) -> List[str]:
    '''Reads the requirements from a file and returns them as a list.'''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements


setup(
    name = 'my_Project',
    version = '0.1.0',
    packages = find_packages(),
    author = 'Sasi kumar',
    author_email = 'sasikumarjnv26@gmail.com',
    description = 'A sample Python project',
    install_requires = get_requirements('requirements.txt')

)