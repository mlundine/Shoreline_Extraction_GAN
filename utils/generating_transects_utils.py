from osgeo import ogr
from shapely.geometry import MultiLineString, LineString, Point
from shapely import wkt
import sys, math
import os

## http://wikicode.wikidot.com/get-angle-of-line-between-two-points
## angle between two points
def getAngle(pt1, pt2):
    x_diff = pt2.x - pt1.x
    y_diff = pt2.y - pt1.y
    return math.degrees(math.atan2(y_diff, x_diff))


## start and end points of chainage tick
## get the first end point of a tick
def getPoint1(pt, bearing, dist):
    angle = bearing + 90
    bearing = math.radians(angle)
    x = pt.x + dist * math.cos(bearing)
    y = pt.y + dist * math.sin(bearing)
    return Point(x, y)


## get the second end point of a tick
def getPoint2(pt, bearing, dist):
    bearing = math.radians(bearing)
    x = pt.x + dist * math.cos(bearing)
    y = pt.y + dist * math.sin(bearing)
    return Point(x, y)


def make_transects(input_path,
                   transect_spacing,
                   transect_length):
    """
    Generates normal transects to an input line shapefile
    inputs:
    input_path: path to shapefile containing the input line
    transect_spacing: distance between each transect in meters
    transect_length: length of each transect in meters
    outputs:
    output_path: path to output shapefile containing transects
    """

    output_path = os.path.splitext(input_path)[0]+'_transects_'+str(transect_spacing)+'m.shp'
    ## set the driver for the data
    driver = ogr.GetDriverByName("Esri Shapefile")
    
    ## open the shapefile in write mode (1)
    ds = driver.Open(input_path)
    shape = ds.GetLayer(0)
    
    ## distance between each points
    distance = transect_spacing
    ## the length of each tick
    tick_length = transect_length

    ## output tick line fc name
    ds_out = driver.CreateDataSource(output_path)
    layer_out = ds_out.CreateLayer('line',shape.GetSpatialRef(),ogr.wkbLineString)

    ## list to hold all the point coords
    list_points = []


    ## distance/chainage attribute
    chainage_fld = ogr.FieldDefn("CHAINAGE", ogr.OFTReal)
    layer_out.CreateField(chainage_fld)
    ## check the geometry is a line
    first_feat = shape.GetFeature(0)

    ln = first_feat
    ## list to hold all the point coords
    list_points = []
    ## set the current distance to place the point
    current_dist = distance
    ## get the geometry of the line as wkt
    line_geom = ln.geometry().ExportToWkt()
    ## make shapely LineString object
    shapely_line = LineString(wkt.loads(line_geom))
    ## get the total length of the line
    line_length = shapely_line.length
    ## append the starting coordinate to the list
    list_points.append(Point(list(shapely_line.coords)[0]))
    ## https://nathanw.net/2012/08/05/generating-chainage-distance-nodes-in-qgis/
    ## while the current cumulative distance is less than the total length of the line
    while current_dist < line_length:
        ## use interpolate and increase the current distance
        list_points.append(shapely_line.interpolate(current_dist))
        current_dist += distance
    ## append end coordinate to the list
    list_points.append(Point(list(shapely_line.coords)[-1]))

    ## add lines to the layer
    ## this can probably be cleaned up better
    ## but it works and is fast!
    for num, pt in enumerate(list_points, 1):
        ## start chainage 0
        if num == 1:
            angle = getAngle(pt, list_points[num])
            line_end_1 = getPoint1(pt, angle, tick_length/2)
            angle = getAngle(line_end_1, pt)
            line_end_2 = getPoint2(line_end_1, angle, tick_length)
            tick = LineString([(line_end_1.x, line_end_1.y), (line_end_2.x, line_end_2.y)])
            feat_dfn_ln = layer_out.GetLayerDefn()
            feat_ln = ogr.Feature(feat_dfn_ln)
            feat_ln.SetGeometry(ogr.CreateGeometryFromWkt(tick.wkt))
            feat_ln.SetField("CHAINAGE", 0)
            layer_out.CreateFeature(feat_ln)

        ## everything in between
        if num < len(list_points) - 1:
            angle = getAngle(pt, list_points[num])
            line_end_1 = getPoint1(list_points[num], angle, tick_length/2)
            angle = getAngle(line_end_1, list_points[num])
            line_end_2 = getPoint2(line_end_1, angle, tick_length)
            tick = LineString([(line_end_1.x, line_end_1.y), (line_end_2.x, line_end_2.y)])
            feat_dfn_ln = layer_out.GetLayerDefn()
            feat_ln = ogr.Feature(feat_dfn_ln)
            feat_ln.SetGeometry(ogr.CreateGeometryFromWkt(tick.wkt))
            feat_ln.SetField("CHAINAGE", distance * num)
            layer_out.CreateFeature(feat_ln)

        ## end chainage
        if num == len(list_points):
            angle = getAngle(list_points[num - 2], pt)
            line_end_1 = getPoint1(pt, angle, tick_length/2)
            angle = getAngle(line_end_1, pt)
            line_end_2 = getPoint2(line_end_1, angle, tick_length)
            tick = LineString([(line_end_1.x, line_end_1.y), (line_end_2.x, line_end_2.y)])
            feat_dfn_ln = layer_out.GetLayerDefn()
            feat_ln = ogr.Feature(feat_dfn_ln)
            feat_ln.SetGeometry(ogr.CreateGeometryFromWkt(tick.wkt))
            feat_ln.SetField("CHAINAGE", int(line_length))
            layer_out.CreateFeature(feat_ln)

    del ds
    return output_path

