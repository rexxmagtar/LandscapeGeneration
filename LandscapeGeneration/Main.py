import bpy
import colorsys

"""Erosion"""
import numpy as np
import util


# Smooths out slopes of `terrain` that are too steep. Rough approximation of the
# phenomenon described here: https://en.wikipedia.org/wiki/Angle_of_repose
def apply_slippage(terrain, repose_slope, cell_width):
    delta = util.simple_gradient(terrain) / cell_width
    smoothed = util.gaussian_blur(terrain, sigma=1.5)
    should_smooth = np.abs(delta) > repose_slope
    result = np.select([np.abs(delta) > repose_slope], [smoothed], terrain)
    return result


def main(terrain, iterationNumber: int):
    # Grid dimension constants
    full_width = len(terrain)
    dim = len(terrain)
    shape = [dim] * 2
    cell_width = full_width / dim
    cell_area = cell_width ** 2

    # Snapshotting parameters. Only needed for generating the simulation
    # timelapse.
    # enable_snapshotting = False
    # my_dir = os.path.dirname(argv[0])
    # snapshot_dir = os.path.join(my_dir, 'sim_snaps')
    # snapshot_file_template = 'sim-%05d.png'
    # if enable_snapshotting:
    #     try:
    #         os.mkdir(snapshot_dir)
    #     except:
    #         pass

    # Water-related constants
    rain_rate = 0.0008 * cell_area
    evaporation_rate = 0.0005

    # Slope constants
    min_height_delta = 0.05
    repose_slope = 0.03
    gravity = 30.0
    gradient_sigma = 0.5

    # Sediment constants
    sediment_capacity_constant = 50.0
    dissolving_rate = 0.25
    deposition_rate = 0.001

    # The numer of iterations is proportional to the grid dimension. This is to
    # allow changes on one side of the grid to affect the other side.
    iterations = iterationNumber

    # `terrain` represents the actual terrain height we're interested in
    # terrain = util.fbm(shape, -2.0)

    # `sediment` is the amount of suspended "dirt" in the water. Terrain will be
    # transfered to/from sediment depending on a number of different factors.
    sediment = np.zeros_like(terrain)

    # The amount of water. Responsible for carrying sediment.
    water = np.zeros_like(terrain)

    # The water velocity.
    velocity = np.zeros_like(terrain)

    for i in range(0, iterations):
        print('%d / %d' % (i + 1, iterations))

        # Add precipitation. This is done by via simple uniform random distribution,
        # although other models use a raindrop model
        water += np.random.rand(*shape) * rain_rate

        # Compute the normalized gradient of the terrain height to determine where
        # water and sediment will be moving.
        gradient = np.zeros_like(terrain, dtype='complex')
        gradient = util.simple_gradient(terrain)
        gradient = np.select([np.abs(gradient) < 1e-10],
                             [np.exp(2j * np.pi * np.random.rand(*shape))],
                             gradient)
        gradient /= np.abs(gradient)

        # Compute the difference between teh current height the height offset by
        # `gradient`.
        neighbor_height = util.sample(terrain, -gradient)
        height_delta = terrain - neighbor_height

        # The sediment capacity represents how much sediment can be suspended in
        # water. If the sediment exceeds the quantity, then it is deposited,
        # otherwise terrain is eroded.
        sediment_capacity = (
                (np.maximum(height_delta, min_height_delta) / cell_width) * velocity *
                water * sediment_capacity_constant)
        deposited_sediment = np.select(
            [
                height_delta < 0,
                sediment > sediment_capacity,
            ], [
                np.minimum(height_delta, sediment),
                deposition_rate * (sediment - sediment_capacity),
            ],
            # If sediment <= sediment_capacity
            dissolving_rate * (sediment - sediment_capacity))

        # Don't erode more sediment than the current terrain height.
        deposited_sediment = np.maximum(-height_delta, deposited_sediment)

        # Update terrain and sediment quantities.
        sediment -= deposited_sediment
        terrain += deposited_sediment
        sediment = util.displace(sediment, gradient)
        water = util.displace(water, gradient)

        # Smooth out steep slopes.
        terrain = apply_slippage(terrain, repose_slope, cell_width)

        # Update velocity
        velocity = gravity * height_delta / cell_width

        # Apply evaporation
        water *= 1 - evaporation_rate

        # Snapshot, if applicable.
        # if enable_snapshotting:
        #     output_path = os.path.join(snapshot_dir, snapshot_file_template % i)
        #     util.save_as_png(terrain, output_path)

    # normalizedterrain = util.normalize(terrain)
    return terrain


"""Perlin noise implementation."""
# Licensed under ISC
from itertools import product
import math
import random


def smoothstep(t):
    """Smooth curve with a zero derivative at 0 and 1, making it useful for
    interpolating.
    """
    return t * t * (3. - 2. * t)


def lerp(t, a, b):
    """Linear interpolation between a and b, given a fraction t."""
    return a + t * (b - a)
    # return (a - b) * ((t * (t * 6.0 - 15.0) + 10.0) * t * t*t) + a


class PerlinNoiseFactory(object):
    """Callable that produces Perlin noise for an arbitrary point in an
    arbitrary number of dimensions.  The underlying grid is aligned with the
    integers.
    There is no limit to the coordinates used; new gradients are generated on
    the fly as necessary.
    """

    def __init__(self, dimension, octaves=1, tile=(), unbias=False):
        """Create a new Perlin noise factory in the given number of dimensions,
        which should be an integer and at least 1.
        More octaves create a foggier and more-detailed noise pattern.  More
        than 4 octaves is rather excessive.
        ``tile`` can be used to make a seamlessly tiling pattern.  For example:
            pnf = PerlinNoiseFactory(2, tile=(0, 3))
        This will produce noise that tiles every 3 units vertically, but never
        tiles horizontally.
        If ``unbias`` is true, the smoothstep function will be applied to the
        output before returning it, to counteract some of Perlin noise's
        significant bias towards the center of its output range.
        """
        self.dimension = dimension
        self.octaves = octaves
        self.tile = tile + (0,) * dimension
        self.unbias = unbias

        # For n dimensions, the range of Perlin noise is ±sqrt(n)/2; multiply
        # by this to scale to ±1
        self.scale_factor = 2 * dimension ** -0.5

        self.gradient = {}

    def _generate_gradient(self):
        # Generate a random unit vector at each grid point -- this is the
        # "gradient" vector, in that the grid tile slopes towards it

        # 1 dimension is special, since the only unit vector is trivial;
        # instead, use a slope between -1 and 1
        if self.dimension == 1:
            return (random.uniform(-1, 1),)

        # Generate a random point on the surface of the unit n-hypersphere;
        # this is the same as a random unit vector in n dimensions.  Thanks
        # to: http://mathworld.wolfram.com/SpherePointPicking.html
        # Pick n normal random variables with stddev 1
        random_point = [random.gauss(0, 1) for _ in range(self.dimension)]
        # Then scale the result to a unit vector
        scale = sum(n * n for n in random_point) ** -0.5
        return tuple(coord * scale for coord in random_point)

    def get_plain_noise(self, *point):
        """Get plain noise for a single point, without taking into account
        either octaves or tiling.
        """
        if len(point) != self.dimension:
            raise ValueError("Expected {} values, got {}".format(
                self.dimension, len(point)))

        # Build a list of the (min, max) bounds in each dimension
        grid_coords = []
        for coord in point:
            min_coord = math.floor(coord)
            max_coord = min_coord + 1
            grid_coords.append((min_coord, max_coord))

        # Compute the dot product of each gradient vector and the point's
        # distance from the corresponding grid point.  This gives you each
        # gradient's "influence" on the chosen point.
        dots = []
        for grid_point in product(*grid_coords):
            if grid_point not in self.gradient:
                self.gradient[grid_point] = self._generate_gradient()
            gradient = self.gradient[grid_point]

            dot = 0
            for i in range(self.dimension):
                dot += gradient[i] * (point[i] - grid_point[i])
            dots.append(dot)

        # Interpolate all those dot products together.  The interpolation is
        # done with smoothstep to smooth out the slope as you pass from one
        # grid cell into the next.
        # Due to the way product() works, dot products are ordered such that
        # the last dimension alternates: (..., min), (..., max), etc.  So we
        # can interpolate adjacent pairs to "collapse" that last dimension.  Then
        # the results will alternate in their second-to-last dimension, and so
        # forth, until we only have a single value left.
        dim = self.dimension
        while len(dots) > 1:
            dim -= 1
            s = smoothstep(point[dim] - grid_coords[dim][0])

            next_dots = []
            while dots:
                next_dots.append(lerp(s, dots.pop(0), dots.pop(0)))

            dots = next_dots

        return dots[0] * self.scale_factor

    def __call__(self, *point):
        """Get the value of this Perlin noise function at the given point.  The
        number of values given should match the number of dimensions.
        """
        ret = 0
        for o in range(self.octaves):
            o2 = 1 << o
            new_point = []
            for i, coord in enumerate(point):
                coord *= o2
                if self.tile[i]:
                    coord %= self.tile[i] * o2
                new_point.append(coord)
            ret += self.get_plain_noise(*new_point) / o2

        # Need to scale n back down since adding all those extra octaves has
        # probably expanded it beyond ±1
        # 1 octave: ±1
        # 2 octaves: ±1½
        # 3 octaves: ±1¾
        ret /= 2 - 2 ** (1 - self.octaves)

        if self.unbias:
            # The output of the plain Perlin noise algorithm has a fairly
            # strong bias towards the center due to the central limit theorem
            # -- in fact the top and bottom 1/8 virtually never happen.  That's
            # a quarter of our entire output range!  If only we had a function
            # in [0..1] that could introduce a bias towards the endpoints...
            r = (ret + 1) / 2
            # Doing it this many times is a completely made-up heuristic.
            for _ in range(int(self.octaves / 2 + 0.5)):
                r = smoothstep(r)
            ret = r * 2 - 1

        return ret


def diamond_square(shape: (int, int),
                   min_height: [float or int],
                   max_height: [float or int],
                   roughness: [float or int],
                   random_seed=None,
                   as_ndarray: bool = True):
    """Runs a diamond square algorithm and returns an array (or list) with the landscape
        An important difference (possibly) between this, and other implementations of the
    diamond square algorithm is how I use the roughness parameter. For each "perturbation"
    I pull a random number from a uniform distribution between min_height and max_height.
    I then take the weighted average between that value, and the average value of the
    "neighbors", whether those be in the diamond or in the square step, as normal. The
    weights used for the weighted sum are (roughness) and (1-roughness) for the random
    number and the average, respectively, where roughness is a float that always falls
    between 0 and 1.
        The roughness value used in each iteration is based on the roughness parameter
    passed in, and is computed as follows:
        this_iteration_roughness = roughness**iteration_number
    where the first iteration has iteration_number = 0. The first roughness value
    actually used (in the very first diamond and square step) is roughness**0 = 1. Thus,
    the values for those first diamond and square step entries will be entirely random.
    This effectively means that I am seeding with A 3x3 grid of random values, rather
    than with just the four corners.
        As the process continues, the weight placed on the random number draw falls from
    the original value of 1, to roughness**1, to roughness**2, and so on, ultimately
    approaching 0. This means that the values of new cells will slowly shift from being
    purely random, to pure averages.
    OTHER NOTES:
    Internally, all heights are between 0 and 1, and are rescaled at the end.
    PARAMETERS
    ----------
    :param shape
        tuple of ints, (int, int): the shape of the resulting landscape
    :param min_height
        Int or Float: The minimum height allowed on the landscape
    :param max_height
        Int or Float: The maximum height allowed on the landscape
    :param roughness
        Float with value between 0 and 1, reflecting how bumpy the landscape should be.
        Values near 1 will result in landscapes that are extremely rough, and have almost no
        cell-to-cell smoothness. Values near zero will result in landscapes that are almost
        perfectly smooth.
        Values above 1.0 will be interpreted as 1.0
        Values below 0.0 will be interpreted as 0.0
    :param random_seed
        Any value. Defaults to None. If a value is given, the algorithm will use it to seed the random
        number generator, ensuring replicability.
    :param as_ndarray
        Bool: whether the landscape should be returned as a numpy array. If set
        to False, the method will return list of lists.
    :returns [list] or nd_array
    """

    # sanitize inputs
    if roughness > 1:
        roughness = 1.0
    if roughness < 0:
        roughness = 0.0

    working_shape, iterations = _get_working_shape_and_iterations(shape)

    # create the array
    diamond_square_array = np.full(working_shape, -1, dtype='float')

    # seed the random number generator
    random.seed(random_seed)

    # seed the corners
    diamond_square_array[0, 0] = random.uniform(0, 1)
    diamond_square_array[working_shape[0] - 1, 0] = random.uniform(0, 1)
    diamond_square_array[0, working_shape[1] - 1] = random.uniform(0, 1)
    diamond_square_array[working_shape[0] - 1, working_shape[1] - 1] = random.uniform(0, 1)

    # do the algorithm
    for i in range(iterations):
        r = math.pow(roughness, i)

        step_size = math.floor((working_shape[0] - 1) / math.pow(2, i))

        _diamond_step(diamond_square_array, step_size, r)
        _square_step(diamond_square_array, step_size, r)

    # rescale the array to fit the min and max heights specified
    diamond_square_array = min_height + (diamond_square_array * (max_height - min_height))

    # trim array, if needed
    final_array = diamond_square_array[:shape[0], :shape[1]]

    if as_ndarray:
        return final_array
    else:
        return final_array.tolist()


def _get_working_shape_and_iterations(requested_shape, max_power_of_two=13):
    """Returns the necessary size for a square grid which is usable in a DS algorithm.
    The Diamond Square algorithm requires a grid of size n x n where n = 2**x + 1, for any
    integer value of x greater than two. To accomodate a requested map size other than these
    dimensions, we simply create the next largest n x n grid which can entirely contain the
    requested size, and return a subsection of it.
    This method computes that size.
    PARAMETERS
    ----------
    requested_shape
        A 2D list-like object reflecting the size of grid that is ultimately desired.
    max_power_of_two
        an integer greater than 2, reflecting the maximum size grid that the algorithm can EVER
        attempt to make, even if the requested size is too big. This limits the algorithm to
        sizes that are manageable, unless the user really REALLY wants to have a bigger one.
        The maximum grid size will have an edge of size  (2**max_power_of_two + 1)
    RETURNS
    -------
    An integer of value n, as described above.
    """
    if max_power_of_two < 3:
        max_power_of_two = 3

    largest_edge = max(requested_shape)

    for power in range(1, max_power_of_two + 1):
        d = (2 ** power) + 1
        if largest_edge <= d:
            return (d, d), power

    # failsafe: no values in the dimensions array were allowed, so print a warning and return
    # the maximum size.
    d = 2 ** max_power_of_two + 1
    print("DiamondSquare Warning: Requested size was too large. Grid of size {0} returned""".format(d))
    return (d, d), max_power_of_two


def _diamond_step(DS_array, step_size, roughness):
    """Does the diamond step for a given iteration.
    During the diamond step, the diagonally adjacent cells are filled:
    Value   None   Value   None   Value  ...
    None   FILLING  None  FILLING  None  ...

    Value   None   Value   None   Value  ...
    ...     ...     ...     ...    ...   ...
    So we'll step with increment step_size over BOTH axes
    """
    # calculate where all the diamond corners are (the ones we'll be filling)
    half_step = math.floor(step_size / 2)
    x_steps = range(half_step, DS_array.shape[0], step_size)
    y_steps = x_steps[:]

    for i in x_steps:
        for j in y_steps:
            if DS_array[i, j] == -1.0:
                DS_array[i, j] = _diamond_displace(DS_array, i, j, half_step, roughness)


def _square_step(DS_array, step_size, roughness):
    """Does the square step for a given iteration.
    During the diamond step, the diagonally adjacent cells are filled:
     Value    FILLING    Value    FILLING   Value   ...
    FILLING   DIAMOND   FILLING   DIAMOND  FILLING  ...

     Value    FILLING    Value    FILLING   Value   ...
      ...       ...       ...       ...      ...    ...
    So we'll step with increment step_size over BOTH axes
    """

    # doing this in two steps: the first, where the every other column is skipped
    # and the second, where every other row is skipped. For each, iterations along
    # the half-steps go vertically or horizontally, respectively.

    # set the half-step for the calls to square_displace
    half_step = math.floor(step_size / 2)

    # vertical step
    steps_x_vert = range(half_step, DS_array.shape[0], step_size)
    steps_y_vert = range(0, DS_array.shape[1], step_size)

    # horizontal step
    steps_x_horiz = range(0, DS_array.shape[0], step_size)
    steps_y_horiz = range(half_step, DS_array.shape[1], step_size)

    for i in steps_x_horiz:
        for j in steps_y_horiz:
            DS_array[i, j] = _square_displace(DS_array, i, j, half_step, roughness)

    for i in steps_x_vert:
        for j in steps_y_vert:
            DS_array[i, j] = _square_displace(DS_array, i, j, half_step, roughness)


def _diamond_displace(DS_array, i, j, half_step, roughness):
    """
    defines the midpoint displacement for the diamond step
    :param DS_array:
    :param i:
    :param j:
    :param half_step:
    :param roughness:
    :return:
    """
    ul = DS_array[i - half_step, j - half_step]
    ur = DS_array[i - half_step, j + half_step]
    ll = DS_array[i + half_step, j - half_step]
    lr = DS_array[i + half_step, j + half_step]

    ave = (ul + ur + ll + lr) / 4.0

    rand_val = random.uniform(0, 1)

    return (roughness * rand_val) + (1.0 - roughness) * ave


def _square_displace(DS_array, i, j, half_step, roughness):
    """
    Defines the midpoint displacement for the square step
    :param DS_array:
    :param i:
    :param j:
    :param half_step:
    :param roughness:
    :return:
    """
    _sum = 0.0
    divide_by = 4

    # check cell "above"
    if i - half_step >= 0:
        _sum += DS_array[i - half_step, j]
    else:
        divide_by -= 1

    # check cell "below"
    if i + half_step < DS_array.shape[0]:
        _sum += DS_array[i + half_step, j]
    else:
        divide_by -= 1

    # check cell "left"
    if j - half_step >= 0:
        _sum += DS_array[i, j - half_step]
    else:
        divide_by -= 1

    # check cell "right"
    if j + half_step < DS_array.shape[0]:
        _sum += DS_array[i, j + half_step]
    else:
        divide_by -= 1

    ave = _sum / divide_by

    rand_val = random.uniform(0, 1)

    return (roughness * rand_val) + (1.0 - roughness) * ave


def create_see():
    vertices = [(0, 0, 0), (0, 100, 0), (100, 100, 0), (100, 0, 0)]
    edges = []
    faces = [(0, 1, 2, 3)]
    flat_mesh = bpy.data.meshes.new('see_mesh')
    flat_mesh.from_pydata(vertices, edges, faces)
    flat_mesh.update()
    # make object from mesh
    new_flat = bpy.data.objects.new('new_see', flat_mesh)
    # make collection
    flat_collection = bpy.data.collections.new('see_see')
    bpy.context.scene.collection.children.link(flat_collection)
    # add object to scene collection
    flat_collection.objects.link(new_flat)

    if flat_mesh.vertex_colors:
        vcol_layer = flat_mesh.vertex_colors.active
    else:
        vcol_layer = flat_mesh.vertex_colors.new()

    for poly in flat_mesh.polygons:
        for loop_index in poly.loop_indices:
            loop_vert_index = flat_mesh.loops[loop_index].vertex_index

            vcol_layer.data[loop_index].color = (0, 0, 1, 0.5)


def drawPlots(Z, title):
    """Draws plots for specified height map"""
    import matplotlib.pyplot as plt
    import numpy as np

    length = len(Z)

    z = np.array([[Z[x, y] for x in range(length)] for y in range(length)])
    x, y = np.meshgrid(range(z.shape[0]), range(z.shape[1]))

    # show hight map in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z)
    plt.title(title + '. 3d height map')
    plt.show()

    # show hight map in 2d
    plt.figure()
    plt.title(title + '. 2d height map')
    p = plt.imshow(z, cmap='Greys')
    plt.colorbar(p)
    plt.show()


def generatePerlinNoize(length, octaves, maxHeight, minHeight):
    perl = PerlinNoiseFactory(2, octaves, tile=(5, 5), unbias=True)

    import numpy as np

    Z = np.zeros((length, length))

    for i in range(length):
        for j in range(length):
            Z[i][j] = minHeight + (perl(i / (length - 1), j / (length - 1))) * (maxHeight - minHeight)

    return Z


def generateTerrain(Z):
    """Generates mesh in blender studio"""
    quadsCount = len(Z) - 1

    length = len(Z);

    map1 = Z

    print("Started")

    vertices = []
    edges = []

    faces = []

    quadLength = 1

    vertices = []

    for i in range(length):
        for j in range(length):
            vertices.append((i * quadLength, j * quadLength, map1[i, j]))

    # TODO: now we have to create Faces based on vertices.

    # for i in range(quadsCount + 1):
    #     for j in range(quadsCount + 1):
    #         vertices.append((i, j, random.randint(0, 0)))
    #         print("added vertices: " + str(i) + ", " + str(j))

    for i in range(((quadsCount + 1) * (quadsCount))):
        if i != 0 and ((i + 1) % (quadsCount + 1) == 0):
            continue
        faces.append([i, i + 1, i + quadsCount + 2, i + quadsCount + 1])
        # print("added face: " + str(faces[-1][0]) + ", " + str(faces[-1][1]) + ", " + str(faces[-1][2]) + ", " + str(
        #     faces[-1][3]))

    flat_mesh = bpy.data.meshes.new('flat_mesh')
    flat_mesh.from_pydata(vertices, edges, faces)
    flat_mesh.update()
    # make object from mesh
    new_flat = bpy.data.objects.new('new_flat', flat_mesh)
    # make collection
    flat_collection = bpy.data.collections.new('flat_collection')
    bpy.context.scene.collection.children.link(flat_collection)
    # add object to scene collection
    flat_collection.objects.link(new_flat)

    maxHeight = max(vertices, key=lambda v: v[2])[2]
    minHeight = 0

    print("Max height = " + str(maxHeight))
    print("Min height = " + str(minHeight))

    dif = maxHeight - minHeight

    if flat_mesh.vertex_colors:
        vcol_layer = flat_mesh.vertex_colors.active
    else:
        vcol_layer = flat_mesh.vertex_colors.new()

    for poly in flat_mesh.polygons:
        for loop_index in poly.loop_indices:
            loop_vert_index = flat_mesh.loops[loop_index].vertex_index
            color = colorsys.hsv_to_rgb(
                min(0.70, (maxHeight - flat_mesh.vertices[loop_vert_index].co[2]) / (dif * 2.4)),
                1, 1)
            vcol_layer.data[loop_index].color = (color[0], color[1], color[2], 0.5)


if __name__ == '__main__':

    #length must be a power of 2 for diamond square algorithm
    length = 2 ** 8;

    max_height = 150;
    min_height = -10;

    map1 = diamond_square(shape=(length, length),
                          min_height=min_height,
                          max_height=max_height,
                          roughness=0.49, random_seed=228)

    drawPlots(map1, 'Diamond algorithm')

    map2 = generatePerlinNoize(length, 4, minHeight=min_height, maxHeight=max_height)

    drawPlots(map2, 'Perlin noise')

    map2 = main(util.normalize(map2), 200)

    map2 = util.normalize(map2, (min_height, max_height))

    drawPlots(map2, 'Perlin noise after erosion')



