import os

def print_directory_tree(root_dir, prefix=''):
    # Print the current directory name
    print(prefix + os.path.basename(root_dir) + '/')
    
    # Get the list of all files and directories in the current directory
    items = os.listdir(root_dir)
    
    # Define directories and files to exclude
    exclude = {'__pycache__', 'ai-core-tools', 'settings', '.gitignore'}
    
    # Filter out hidden files and directories, and excluded items
    items = [item for item in items if not item.startswith('.') and item not in exclude]
    
    # Loop through all items in the current directory
    for index, item in enumerate(items):
        item_path = os.path.join(root_dir, item)
        
        # Determine the appropriate prefix for child items
        if index == len(items) - 1:
            new_prefix = prefix + '    '
            item_prefix = prefix + '└── '
        else:
            new_prefix = prefix + '│   '
            item_prefix = prefix + '├── '
        
        # Print the item
        if os.path.isdir(item_path):
            print(item_prefix + item + '/')
            print_directory_tree(item_path, new_prefix)
        else:
            print(item_prefix + item)

# Set the root directory you want to print the structure of
root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print_directory_tree(root_directory)
