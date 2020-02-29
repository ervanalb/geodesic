import solid
import solid.utils
import geodesic as geo
import numpy as np

def sphere():
    """ Creates a solid geodesic sphere """
    (vs, es, fs) = geo.sphere(v=3)
    return solid.polyhedron(vs, fs)

def ball_and_stick():
    """ Creates a ball-and-stick model of a geodesic sphere """

    (vs, es, fs) = geo.sphere(v=4)
    vms = geo.vertex_matrices(vs)
    ems = geo.edge_matrices(es, vs)
    els = geo.edge_lengths(es, vs)

    def shape_at_vertex(m):
        return solid.multmatrix(m)(solid.sphere(r=0.05))

    def shape_at_edge(m, el):
        return solid.multmatrix(m)(
                solid.utils.rot_z_to_right(
                    solid.cylinder(r1=0.03, r2=0.03, h=el, center=True)
                )
            )

    shape = solid.union()(*[shape_at_vertex(m) for m in vms],
                          *[shape_at_edge(m, el) for m, el in zip(ems, els)])

    return shape

def golfball():
    """ Creates a golfball by putting dimples at vertices """

    (vs, es, fs) = geo.sphere(v=5)
    vms = geo.vertex_matrices(vs)

    def shape_at_vertex(m):
        return solid.multmatrix(m)(solid.utils.up(0.09)(solid.sphere(r=0.12)))

    shape = solid.difference()(
        solid.sphere(r=1, segments=FN * 4),
        solid.union()(*[shape_at_vertex(m) for m in vms])
    )

    return shape

def spaceship_earth():
    """ Creates a model of EPCOT's spaceship earth """

    (vs, es, fs) = geo.sphere(v=8)
    fms = geo.face_matrices(fs, vs)
    fts = geo.face_triangles_2d(fs, vs, fms)

    def shape_at_face(m, ft):
        center = np.array([0, 0])
        panel1 = np.vstack((ft[0], ft[1], center))
        panel2 = np.vstack((ft[1], ft[2], center))
        panel3 = np.vstack((ft[2], ft[0], center))

        panel_angle = 20

        def panel_shape(panel_tri):
            edge_midpoint = 0.5 * (panel_tri[0] + panel_tri[1])
            rotation_vector = panel_tri[1] - panel_tri[0]

            transform = lambda x: solid.translate(edge_midpoint)(solid.rotate(a=panel_angle, v=rotation_vector)(solid.translate(-edge_midpoint)(x)))

            return transform(solid.linear_extrude(0.01)(solid.polygon(panel_tri * 0.9)))

        return solid.multmatrix(m)(
            solid.union()(*[panel_shape(p) for p in (panel1, panel2, panel3)])
            )

    inner_radius = np.mean(np.linalg.norm(np.array(fms)[:, 0:3, 3], axis=1))

    return solid.union()(solid.sphere(r=inner_radius, segments=FN * 4), *[shape_at_face(m, ft) for m, ft in zip(fms, fts)])

def tetrahedron():
    """ Creates a ball-and-stick model of a geodesic sphere based off of an tetrahedron,
    highlighting vertices of valence 3 """

    (vs, es, fs) = geo.sphere(v=3, base=geo.tetrahedron_vertices)
    vms = geo.vertex_matrices(vs)
    ems = geo.edge_matrices(es, vs)
    els = geo.edge_lengths(es, vs)
    ves = geo.vertex_edges(es)

    highlight_valence = 3

    def shape_at_vertex(m, ve):
        shape = solid.sphere(r=0.05)
        if len(ve) == highlight_valence:
            shape = solid.color("red")(shape)
        return solid.multmatrix(m)(shape)

    def shape_at_edge(m, el, e):
        shape = solid.cylinder(r1=0.03, r2=0.03, h=el, center=True)
        if any(len(ves[i]) == highlight_valence for i in e):
            shape = solid.color("red")(shape)

        return solid.multmatrix(m)(
                solid.utils.rot_z_to_right(
                    shape
                )
            )

    shape = solid.union()(*[shape_at_vertex(m, ve) for m, ve in zip(vms, ves)],
                          *[shape_at_edge(m, el, e) for m, el, e in zip(ems, els, es)])

    return shape

def pointy_dome():
    """ Creates a geodesic dome with the vertices pushed in little bit to make it pointy """

    (vs, es, fs) = geo.sphere(v=4)

    (vs, es, fs) = geo.filter_vertices(lambda pt: pt[2] > -0.3, vs, es, fs)

    vs[:, 0:2] = vs[:, 0:2] * (1. - 0.5 * vs[:, 2])[:, None]

    vms = geo.vertex_matrices(vs)
    ems = geo.edge_matrices(es, vs)
    els = geo.edge_lengths(es, vs)

    def shape_at_vertex(m):
        return solid.multmatrix(m)(solid.sphere(r=0.05))

    def shape_at_edge(m, el):
        return solid.multmatrix(m)(
                solid.utils.rot_z_to_right(
                    solid.cylinder(r1=0.03, r2=0.03, h=el, center=True)
                )
            )

    shape = solid.union()(*[shape_at_vertex(m) for m in vms],
                          *[shape_at_edge(m, el) for m, el in zip(ems, els)])

    return shape

FN = 15

shapes = solid.union()(
    solid.utils.right(0)(sphere()),
    solid.utils.right(3)(ball_and_stick()),
    solid.utils.right(6)(golfball()),
    solid.utils.right(9)(spaceship_earth()),
    solid.utils.right(12)(tetrahedron()),
    solid.utils.right(15)(pointy_dome()),
)

solid.scad_render_to_file(shapes, 'example.scad', file_header = '$fn = {};\n'.format(FN))
