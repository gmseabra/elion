# ORM database design
from sqlalchemy import (String, Integer, ForeignKey, MetaData, Boolean, BigInteger)
from nas_storage_app import db
print('models start...')
convention = {
    "ix": '%(column_0_label)s_ix',
    "uq": "%(table_name)s_%(column_0_name)s_UNIQUE",
    # "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(column_0_name)s_%(table_name)s",
    # "pk": "pk_%(table_name)s"
    }

metadata = MetaData(naming_convention=convention)


class SourceData(db.Model):
    __tablename__ = 'source_data'

    id = db.Column('id', Integer, primary_key=True, autoincrement=True, index=True,
                   nullable=False, unique=True)
    md5checksum = db.Column('md5checksum', String(length=36))

    def __init__(self, md5checksum):
        self.md5checksum = md5checksum


class DataToSite(db.Model):
    __tablename__ = 'data_to_site'
    id = db.Column('id', Integer, primary_key=True, autoincrement=True)

    source_data_id = db.Column('source_data_id', Integer, ForeignKey('source_data.id'))
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'), 
                           nullable=True)
    site_number = db.Column('site_number', Integer)

    def __init__(self, source_data_id, dataset_id, site_number):
        self.source_data_id = source_data_id
        self.dataset_id = dataset_id
        self.site_number = site_number


class DatasetIndex(db.Model):
    __tablename__ = 'dataset_index'

    id = db.Column('id', Integer, ForeignKey('datasets.id'), primary_key=True)
    filepath = db.Column('filepath', String(length=5000), nullable=True)
    indexed = db.Column('indexed', Boolean, nullable=False)
    sensor_id = db.Column('sensor_id', ForeignKey('sensor_id_index.id'))
    sensor_loc_id = db.Column('sensor_loc_id', ForeignKey('sensor_loc_id_index.id'))
    type_id = db.Column('type_id', Integer, ForeignKey('dataset_type_id_index.type_id'))
    md5checksum_filename = db.Column('md5checksum_filename', String(length=36))

    def __init__(self, filepath, indexed, sensor_id, sensor_loc_id, type_id,
                 md5checksum_filename):
        self.filepath = filepath
        self.indexed = indexed
        self.sensor_id = sensor_id
        self.sensor_loc_id = sensor_loc_id
        self.type_id = type_id
        self.md5checksum_filename = md5checksum_filename


class SourceDataIndex(db.Model):
    __tablename__ = 'source_data_index'

    source_data_index_id = db.Column('source_data_index_id', Integer, ForeignKey('source_data.id'), primary_key=True)
    filepath = db.Column('filepath', String(length=5000), nullable=True)
    type_id = db.Column('type_id', Integer, ForeignKey('source_data_type_id_index.type_id'))
    source_data_index_data_collection_kit_id = db.Column('source_data_index_data_collection_kit_id', Integer, ForeignKey('data_collection_kit_id.id'))
    file_size = db.Column('file_size', BigInteger)

    def __init__(self, source_data_index_id, filepath, type_id, source_data_index_data_collection_kit_id, file_size):
        self.source_data_index_id = source_data_index_id
        self.filepath = filepath
        self.type_id = type_id
        self.source_data_index_data_collection_kit_id = source_data_index_data_collection_kit_id
        self.file_size = file_size


class DatasetTypeIdIndex(db.Model):
    __tablename__ = 'dataset_type_id_index'

    type_id = db.Column('type_id', Integer, nullable=False, autoincrement=True,
                        unique=True, primary_key=True)
    type = db.Column('type', String(length=45))

    def __init__(self, type):
        self.type = type


class SourceDataTypeIdIndex(db.Model):
    __tablename__ = 'source_data_type_id_index'

    type_id = db.Column('type_id', Integer, nullable=False, autoincrement=True,
                        unique=True, primary_key=True)
    type = db.Column('type', String(length=45))

    def __init__(self, type):
        self.type = type


class Datasets(db.Model):
    __tablename__ = 'datasets'

    id = db.Column('id', Integer, primary_key=True, autoincrement=True)
    md5checksum_frame_0 = db.Column('md5checksum_frame_0', String(length=36))

    def __init__(self, md5checksum_frame_0):
        self.md5checksum_frame_0 = md5checksum_frame_0


class SourceDataToDataset(db.Model):
    __tablename__ = 'source_data_to_dataset'

    id = db.Column('id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    source_data_id = db.Column('source_data_id', Integer, ForeignKey('source_data.id'))

    def __init__(self, dataset_id, source_data_id):
        self.dataset_id = dataset_id
        self.source_data_id = source_data_id


class IndexDataStore(db.Model):
    __tablename__ = 'index_data_store'

    id = db.Column('id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    directory = db.Column('directory', String(length=5000), nullable=True)

    def __init__(self, dataset_id, directory):
        self.dataset_id = dataset_id
        self.directory = directory


class LabeledDataStore(db.Model):
    __tablename__ = 'labeled_data_store'

    id = db.Column('id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    directory = db.Column('directory', String(length=5000), nullable=True)

    def __init__(self, dataset_id, directory):
        self.dataset_id = dataset_id
        self.directory = directory


class DataCollectionKitId(db.Model):
    __tablename__ = 'data_collection_kit_id'

    id = db.Column('id', Integer, primary_key=True, autoincrement=True)
    md5checksum = db.Column('md5checksum', String(length=36))
    source_data_id = db.Column('source_data_id', ForeignKey('source_data.id'))
    dataset_id = db.Column('dataset_id', ForeignKey('datasets.id'))

    def __init__(self, md5checksum, source_data_id, dataset_id):
        self.md5checksum = md5checksum
        self.source_data_id = source_data_id
        self.dataset_id = dataset_id


class SensorIdIndex(db.Model):
    __tablename__ = 'sensor_id_index'

    id = db.Column('id', Integer, primary_key=True, autoincrement=True)
    type = db.Column('type', String(length=45))

    def __init__(self, type):
        self.type = type


class SensorLocIdIndex(db.Model):
    __tablename__ = 'sensor_loc_id_index'
    id = db.Column('id', Integer, primary_key=True, autoincrement=True)
    location = db.Column('location', String(length=45))

    def __init__(self, location):
        self.location = location


class DatasetStagingStatus(db.Model):
    __tablename__ = 'dataset_staging_status'

    id = db.Column('id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    staged = db.Column('staged', Boolean, default=False)
    labeled = db.Column('labeled', Boolean, default=False)

    def __init__(self, dataset_id, staged, labeled):
        self.dataset_id = dataset_id
        self.staged = staged
        self.labeled = labeled


class IndexAttempted(db.Model):
    __tablename__ = 'index_attempted'

    id = db.Column('id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    model_name = db.Column('model_name', String(length=45))

    def __init__(self, dataset_id, model_name):
        self.dataset_id = dataset_id
        self.model_name = model_name


class GtagTypeIdIndex(db.Model):
    __tablename__ = 'gtag_type_id_index'

    type_id = db.Column('type_id', Integer, nullable=False, autoincrement=True,
                        unique=True, primary_key=True)
    type = db.Column('type', String(length=45))

    def __init__(self, type):
        self.type = type


class GtagType(db.Model):
    __tablename__ = 'gtag_type'

    id = db.Column('id', Integer, primary_key=True, autoincrement=True, index=True,
                   nullable=False, unique=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey(
        'datasets.id'), nullable=False, index=True)
    type_id = db.Column('type_id', Integer, ForeignKey('gtag_type_id_index.type_id'),
                        nullable=False, index=True)

    def __init__(self, source_data_id, type_id):
        self.source_data_id = source_data_id
        self.type_id = type_id


# TODO: remove underscore
class RosInfo(db.Model):
    __tablename__ = 'ros_info'

    ros_id = db.Column('ros_id', Integer, primary_key=True, autoincrement=True)
    file_extension = db.Column('file_extension', String(length=45))
    filepath = db.Column('filepath', String(length=5000), nullable=True)
    version = db.Column('version', String(length=45))
    duration = db.Column('duration', String(length=45))
    ros_start = db.Column('ros_start', String(length=45))
    ros_end = db.Column('ros_end', String(length=45))
    ros_size = db.Column('ros_size', String(length=45))
    ros_messages = db.Column('ros_messages', String(length=45))
    ros_compression = db.Column('ros_compression', String(length=45))
    ros_types = db.Column('ros_types', String(length=45))
    ros_topics = db.Column('ros_topics', String(length=45))

    def __init__(self, file_extension, filepath, version, duration, ros_start,
                 ros_end, ros_size, ros_messages, ros_compression, ros_types, ros_topics):
        self.file_extension = file_extension
        self.filepath = filepath
        self.version = version
        self.duration = duration
        self.ros_start = ros_start
        self.ros_end = ros_end
        self.ros_size = ros_size
        self.ros_messages = ros_messages
        self.ros_compression = ros_compression
        self.ros_types = ros_types
        self.ros_topics = ros_topics


class RosTypes(db.Model):
    __tablename__ = 'ros_types'

    types_id = db.Column('types_id', Integer, primary_key=True, autoincrement=True)
    types_name = db.Column('types_name', String(length=45))
    types_code = db.Column('types_code', String(length=45))

    def __init__(self, types_name, types_code):
        self.types_name = types_name
        self.types_code = types_code


class RosTopics(db.Model):
    __tablename__ = 'ros_topics'

    topics_id = db.Column('topics_id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    ros_topic_dir = db.Column('ros_topic_dir', String(length=5000))

    def __init__(self, dataset_id, ros_topic_dir):
        self.dataset_id = dataset_id
        self.ros_topic_dir = ros_topic_dir


class ConfigTable(db.Model):
    __tablename__ = 'config_table'

    image_id = db.Column('image_id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    source_data_id = db.Column('source_data_id', Integer, ForeignKey('source_data.id'))
    configversion = db.Column('configversion', String(length=45))
    ptag_version = db.Column('ptag_version', String(length=45))

    def __init__(self, dataset_id, source_data_id, configversion, ptag_version):
        self.dataset_id = dataset_id
        self.source_data_id = source_data_id
        self.configversion = configversion
        self.ptag_version = ptag_version


class CustomerInfoTable(db.Model):
    __tablename__ = 'customer_info_table'

    image_id = db.Column('image_id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    source_data_id = db.Column('source_data_id', Integer, ForeignKey('source_data.id'))
    collection_date = db.Column('collection_date', String(length=45))
    customer = db.Column('customer', String(length=45))
    country = db.Column('country', String(length=45))
    province = db.Column('province', String(length=45))
    city = db.Column('city', String(length=45))
    gps = db.Column('gps', String(length=45))
    work_site = db.Column('work_site', String(length=45))

    def __init__(self, dataset_id, source_data_id, collection_date, customer, country,
                 province, city, gps, work_site):
        self.dataset_id = dataset_id
        self.source_data_id = source_data_id
        self.collection_date = collection_date
        self.customer = customer
        self.country = country
        self.province = province
        self.city = city
        self.gps = gps
        self.work_site = work_site


class MachineInfoTable(db.Model):
    __tablename__ = 'machine_info_table'

    image_id = db.Column('image_id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    source_data_id = db.Column('source_data_id', Integer, ForeignKey('source_data.id'))
    model = db.Column('model', String(length=45))
    serial_number = db.Column('serial_number', String(length=45))
    sensor_types = db.Column('sensor_types', String(length=45))
    restricted_uses = db.Column('restricted_uses', String(length=45))
    sensor_location = db.Column('sensor_location', String(length=45))

    def __init__(self, dataset_id, source_data_id, model, serial_number, sensor_types, restricted_uses,
                 sensor_location):
        self.dataset_id = dataset_id
        self.source_data_id = source_data_id
        self.model = model
        self.serial_number = serial_number
        self.sensor_types = sensor_types
        self.restricted_uses = restricted_uses
        self.sensor_location = sensor_location


class ProjectInfoTable(db.Model):
    __tablename__ = 'project_info_table'

    image_id = db.Column('image_id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    source_data_id = db.Column('source_data_id', Integer, ForeignKey('source_data.id'))
    application_type = db.Column('application_type', String(length=45))
    team = db.Column('team', String(length=45))
    project = db.Column('project', String(length=45))
    license_agreement = db.Column('license_agreement', String(length=45))

    def __init__(self, dataset_id, source_data_id, application_type, team, project, license_agreement):
        self.dataset_id = dataset_id
        self.source_data_id = source_data_id
        self.application_type = application_type
        self.team = team
        self.project = project
        self.license_agreement = license_agreement


# create tables for can message from B7, Ethernet data, detect stream, s cam, b cam, radar,
# The following data comes from s cam detect stream
class PacketInfoTable(db.Model):
    __tablename__ = 'packet_info_table'

    image_id = db.Column('image_id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    source_data_id = db.Column('source_data_id', Integer, ForeignKey('source_data.id'))
    cam_location = db.Column('cam_location', String(length=45))
    protocol_version = db.Column('protocol_version', String(length=45))
    packet_type = db.Column('packet_type', String(length=45))
    serial_number = db.Column('serial_number', String(length=45))

    def __init__(self, dataset_id, source_data_id, cam_location, protocol_version, packet_type, serial_number):
        self.dataset_id = dataset_id
        self.source_data_id = source_data_id
        self.cam_location = cam_location
        self.protocol_version = protocol_version
        self.packet_type = packet_type
        self.serial_number = serial_number


class ObjectDetectTable(db.Model):
    __tablename__ = 'object_detect_table'

    image_id = db.Column('image_id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    source_data_id = db.Column('source_data_id', Integer, ForeignKey('source_data.id'))
    cam_location = db.Column('cam_location', String(length=45))
    number_of_objects = db.Column('number_of_objects', String(length=45))
    target_x_distance = db.Column('target_x_distance', String(length=45))
    target_y_distance = db.Column('target_y_distance', String(length=45))
    target_centroid_pixel_column = db.Column('target_centroid_pixel_column', String(length=45))
    target_centroid_pixel_row = db.Column('target_centroid_pixel_row', String(length=45))
    target_ground_point_pixel_row = db.Column('target_ground_point_pixel_row', String(length=45))
    target_ground_point_pixel_column = db.Column('target_ground_point_pixel_column', String(length=45))
    object_classification = db.Column('object_classification', String(length=45))
    bearing_angle = db.Column('bearing_angle', String(length=45))
    object_status = db.Column('object_status', String(length=45))

    def __init__(self, dataset_id, source_data_id, cam_location, number_of_objects, target_x_distance,
                 target_y_distance, target_centroid_pixel_column,
                 target_centroid_pixel_row, target_ground_point_pixel_row,
                 target_ground_point_pixel_column, bearing_angle, object_status):

        self.dataset_id = dataset_id
        self.source_data_id = source_data_id
        self.cam_location = cam_location
        self.number_of_objects = number_of_objects
        self.target_x_distance = target_x_distance
        self.target_y_distance = target_y_distance
        self.target_centroid_pixel_column = target_centroid_pixel_column
        self.target_centroid_pixel_row = target_centroid_pixel_row
        self.target_ground_point_pixel_row = target_ground_point_pixel_row
        self.target_ground_point_pixel_column = target_ground_point_pixel_column
        self.bearing_angle = bearing_angle
        self.object_status = object_status


class VelocityComponentTable(db.Model):
    __tablename__ = 'velocity_component_table'

    image_id = db.Column('image_id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    source_data_id = db.Column('source_data_id', Integer, ForeignKey('source_data.id'))
    cam_location = db.Column('cam_location', String(length=45))

    bounding_box_height = db.Column('bounding_box_height', String(length=45))
    bounding_box_width = db.Column('bounding_box_width', String(length=45))
    bounding_box_left_cornerX = db.Column('bounding_box_left_cornerX', String(length=45))
    bounding_box_left_cornerY = db.Column('bounding_box_left_cornerY', String(length=45))
    target_north_velocity = db.Column('target_north_velocity', String(length=45))
    target_north_acceleration = db.Column('target_north_acceleration', String(length=45))
    target_E_location_tracker_distanceX = db.Column('target_E_location_tracker_distanceX', String(length=45))

    def __init__(self, dataset_id, source_data_id, cam_location, bounding_box_height,
                 bounding_box_width, bounding_box_left_cornerX, bounding_box_left_cornerY,
                 object_classification, target_north_velocity, target_north_acceleration,
                 target_E_location_tracker_distanceX):

        self.dataset_id = dataset_id
        self.source_data_id = source_data_id
        self.cam_location = cam_location
        self.bounding_box_height = bounding_box_height
        self.bounding_box_width = bounding_box_width
        self.bounding_box_left_cornerX = bounding_box_left_cornerX
        self.bounding_box_left_cornerY = bounding_box_left_cornerY
        self.object_classification = object_classification
        self.target_north_velocity = target_north_velocity
        self.target_north_acceleration = target_north_acceleration
        self.target_E_location_tracker_distanceX = target_E_location_tracker_distanceX


class GPSLocationInfoTable(db.Model):
    __tablename__ = 'gps_location_info_table'

    image_id = db.Column('image_id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    source_data_id = db.Column('source_data_id', Integer, ForeignKey('source_data.id'))

    latitude = db.Column('latitude', String(length=45))
    longitude = db.Column('longitude', String(length=45))
    alt_hae = db.Column('alt_hae', String(length=45))
    alt_msl = db.Column('alt_msl', String(length=45))
    gps_capture_time = db.Column('gps_capture_time', String(length=45))

    def __init__(self, dataset_id, source_data_id, latitude, longitude, alt_hae, alt_msl, gps_capture_time):

        self.dataset_id = dataset_id
        self.source_data_id = source_data_id
        self.latitude = latitude
        self.longitude = longitude
        self.alt_hae = alt_hae
        self.alt_msl = alt_msl
        self.gps_capture_time = gps_capture_time


class GPSMachineInfoTable(db.Model):
    __tablename__ = 'gps_machine_info_table'

    image_id = db.Column('image_id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    source_data_id = db.Column('source_data_id', Integer, ForeignKey('source_data.id'))

    speed = db.Column('speed', String(length=45))
    climb = db.Column('climb', String(length=45))

    def __init__(self, dataset_id, source_data_id, speed, climb):

        self.dataset_id = dataset_id
        self.source_data_id = source_data_id
        self.latitude = speed
        self.longitude = climb


class CocoAnnotationTable(db.Model):
    __tablename__ = 'coco_annotation_table'

    annotation_id = db.Column('annotation_id', Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column('dataset_id', Integer, ForeignKey('datasets.id'))
    category_id = db.Column('category_id', Integer)
    segmentation = db.Column('segmentation', String(length=45))
    area = db.Column('area', String(length=45))
    bbox = db.Column('bbox', String(length=45))
    iscrowd = db.Column('iscrowd', String(length=45))
    attributes = db.Column('attributes', String(length=45))

    def __init__(self, annotation_id, dataset_id, category_id, segmentation, area, bbox,
                 iscrowd, attributes):

        self.annotation_id = annotation_id
        self.dataset_id = dataset_id
        self.category_id = category_id
        self.segmentation = segmentation
        self.area = area
        self.bbox = bbox
        self.iscrowd = iscrowd
        self.attributes = attributes
