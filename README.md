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
Project "LandscapeGeneration" contains all files needed for testing. Just run "Main.py" file for example demonstration.
Notice that if you want to use blender integration you will also need to put necessary python libs in Blender_root_path\startup folder.

# Some beautiful examples
![image](https://user-images.githubusercontent.com/51932532/116790947-c83f5000-aabf-11eb-92ee-74e2e0da7dd5.png)
![image](https://user-images.githubusercontent.com/51932532/116795903-e9646880-aae0-11eb-8ae7-120654c25aeb.png)
![image](https://user-images.githubusercontent.com/51932532/116811229-cec9d800-ab50-11eb-8002-6cbdb0b79067.png)
![image](https://user-images.githubusercontent.com/51932532/116817652-20ce2600-ab70-11eb-87d1-dd5f6837b1e2.png)
![image](https://user-images.githubusercontent.com/51932532/116817844-18c2b600-ab71-11eb-97a3-9c2290613959.png)
![image](https://user-images.githubusercontent.com/51932532/116818154-858a8000-ab72-11eb-983c-10ad36e13aa6.png)
![image](https://user-images.githubusercontent.com/51932532/116818341-95ef2a80-ab73-11eb-9385-8877f3a72d15.png)

Песчаные степи без эрозии.
![image](https://user-images.githubusercontent.com/51932532/117538517-3f6c6b00-b00f-11eb-9d0b-830d8fe26f00.png)
Песчаные степи после эрозии.
![image](https://user-images.githubusercontent.com/51932532/117538699-0bde1080-b010-11eb-9ac2-55e27d381b2a.png)
![image](https://user-images.githubusercontent.com/51932532/117702836-b3ef0780-b1d1-11eb-9199-36e9949be74e.png)
![image](https://user-images.githubusercontent.com/51932532/117703046-f7e20c80-b1d1-11eb-8890-20d25e1a99d0.png)
![image](https://user-images.githubusercontent.com/51932532/117793382-d70ecb00-b254-11eb-9e43-2f8d7573176e.png)
![image](https://user-images.githubusercontent.com/51932532/117794128-9cf1f900-b255-11eb-9cf4-00330b8e020f.png)





