Example from [here](https://github.com/wolfpld/tracy/tree/master/examples/OpenCLVectorAdd) that was slightly modified so that it works outside the `cmake` infrastructure of `tracy`

First install `tracy` and launch it. Then check the version you have, say it is `v0.11.1`.

To use it, clone `tracy` next to the `LINMA2710` folder:
```sh
$ git clone https://github.com/wolfpld/tracy.git
$ git checkout v0.11.1 # To use the same version
```
You can also clone it elsewhere but then you need to modify the `TRACY_DIR` variable in this `Makefile`.
You need to clone it because we need the files in the `public` subfolder.

Then, go back in this folder and do
```sh
$ make
```
Now, click on "Connect" on the tracy application. Then run:
```sh
$ ./OpenCLVectorAdd
```
You should now see the trace in the tracy application.
