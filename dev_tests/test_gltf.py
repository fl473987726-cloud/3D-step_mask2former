from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.RWGltf import RWGltf_CafWriter
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFApp import XCAFApp_Application
from OCP.TCollection import TCollection_ExtendedString, TCollection_AsciiString
from OCP.TColStd import TColStd_IndexedDataMapOfStringString
from OCP.Message import Message_ProgressRange
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_LabelSequence

def convert_step_to_gltf(step_file, gltf_file):
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    
    reader = STEPCAFControl_Reader()
    status = reader.ReadFile(step_file)
    if status != 1:
        print("Failed to read")
        return
    reader.Transfer(doc)
    
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(labels)
    for i in range(1, labels.Length() + 1):
        shape = shape_tool.GetShape_s(labels.Value(i))
        BRepMesh_IncrementalMesh(shape, 0.1)
    
    writer = RWGltf_CafWriter(TCollection_AsciiString(gltf_file), True)
    m = TColStd_IndexedDataMapOfStringString()
    writer.Perform(doc, m, Message_ProgressRange())
    print("Exported GLTF")

if __name__ == "__main__":
    convert_step_to_gltf(r"e:\aaaa-WUT\lw\ASCCAD\test_step\colored\01_closed_pocket.step", "test.gltf")
