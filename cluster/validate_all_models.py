import subprocess
import glob

output_dir = "../models"
save_result_dir = "../results"

for f in glob.glob(output_dir+"/*.pth"):

    # split filename
    values = f.split("-")

    # Get the values we need
    expose_stack = values[6] == "True"
    without_partition = values[8] == "True"
    reduced = values[9] == "True"
    seed = values[2]

    if len(values) > 17:
        without_save_load_partition = values[17].split(".")[0].lower() == "true"
    else:
        without_save_load_partition = False

    # Get correct operations
    operations="none"
    if reduced:
        operations="reduced"
    elif without_partition:
        operations="no-partition"
    elif without_save_load_partition:
        operations="no-save-load-partition"

    # Job name
    name="{}-{}-{}-{}-{}".format(seed, without_partition, reduced, without_save_load_partition, operations)

    # Generate the command
    f = f.replace("\n", "")
    command = "bash submit_validate.sh {} {} {} {}".format(
        f, operations, name, save_result_dir
        )
    print(command)

    # execute the command
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)