import pymeshfix as mf
import trimesh.repair as repair

def exception_catcher(func):
    def new_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"ERR_{func.__name__} for shape {args[0]['meta_data']['name']}")
            return {f"ERR_{func.__name__}": (str(type(e)), str(e))}

    return new_func


def fill_holes_old(mesh):
    # https://pymeshfix.pyvista.org/index.html: M. Attene. A lightweight approach to repairing digitized polygon meshes. The Visual Computer, 2010. (c) Springer. DOI: 10.1007/s00371-010-0416-3
    meshfix = mf.MeshFix(mesh)
    meshfix.repair(verbose=True)
    repaired = meshfix.mesh
    return repaired

def fill_holes(mesh):
    return mesh.fill_holes(1000)

