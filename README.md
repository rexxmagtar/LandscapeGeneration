# Realistic landscape generation



Main aim of this project is to research different methods for realistic landscape generation.



## Short description

In projects terms, landscape generation means generation of a 2D heights map. So all algorithms provide a n x m matrix as a result. All calculations are done using python. Fast visualisation is also done from the scratch. Project also provides methods for landscape visualisation in "Blender studio". It is always a huge time-consuming task, though it provides better visualization.



# Algorithms

So far the following algorithms were implemented:

## Noise algorithm (Perlin noise)

Classic noise algorithm for generation smooth fractal landscape.

[Perlin noise info](https://en.wikipedia.org/wiki/Perlin_noise#:~:text=Perlin%20noise%20is%20a%20procedural,details%20are%20the%20same%20size)

Results:



![image](https://user-images.githubusercontent.com/51932532/115124691-32d69300-9fcc-11eb-8ab4-3a69b72995f3.png)

![image](https://user-images.githubusercontent.com/51932532/115124776-c6a85f00-9fcc-11eb-93af-726ff159564d.png)



## Diamond algorithm 

Another fractal algorithm that provides much more flexibility and realism to generated landscapes.

[Classic algorithm info](https://en.wikipedia.org/wiki/Diamond-square_algorithm)

Results:


![image](https://user-images.githubusercontent.com/51932532/115124970-fa37b900-9fcd-11eb-8d51-9f3db7becd29.png)

![image](https://user-images.githubusercontent.com/51932532/115125078-982b8380-9fce-11eb-8494-e9cc22b9ad5a.png)



## Hydraulic erosion

Real world hydraulic  erosion was simulated. A lot of arguments such as sediment, evaporation, rain rate and e.t.c can be configured.

Erosion can be applied to already configured landscape only, overwise, no result could be achieved. So some noise, for example, can be used to prepare terrain for the erosion.



Results:

Erosion applied to terrain generated using diamond algorithm.

![image](https://user-images.githubusercontent.com/51932532/115125323-0de41f00-9fd0-11eb-9b7d-b37dfa8118ba.png)



Landscape before erosion:

![image](https://user-images.githubusercontent.com/51932532/115125833-7680cb00-9fd3-11eb-8085-1e98eeb7c7ad.png)



Landscape after erosion:

![image](https://user-images.githubusercontent.com/51932532/115125837-800a3300-9fd3-11eb-8363-181db604de1b.png)

# Usage
Project "FractalGeneration" contains all files needed for testing. Just run "Main.py" file for example demonstration.
Notice that if you want to use blender integration you will also need to put necessary python libs in Blender_root_path\startup folder. Dummy Blender project for testing is already created and is called "testDiamond.blend".
