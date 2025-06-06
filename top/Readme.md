# Code Structure

* Wearable_modules: This module contains all the necessary components required to execute the `wearable.py` file.
* Edge_module: This module includes the `edge.py` file, which is utilized to run the `wearable_edge.py` script.
* Server_modules: This module comprises the `server.py` file, which supports the execution of the `wearable_server.py` script.

Please ensure that all files are located within the same folder or directory.

* The `wearable.py` file depends on the contents of the `Wearable_modules`.
* The `wearable_edge.py` file requires both the `Edge_module` files and the `wearable.py` file.
* The `wearable_server.py` file depends on the `Server_modules`, as well as on both `wearable_edge.py` and `wearable.py` files.

