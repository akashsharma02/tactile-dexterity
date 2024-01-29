import numpy as np
from holodex.utils.files import get_path_in_package, get_yaml_data
from ikpy import chain


class AllegroKDL(object):
    def __init__(self):
        # Getting the URDF path
        urdf_path = get_path_in_package("robot/assets/allegro_hand_right.urdf")

        # Loading Allegro Hand configs
        self.hand_configs = get_yaml_data(
            get_path_in_package("robot/configs/allegro_info.yaml")
        )
        self.finger_configs = get_yaml_data(
            get_path_in_package("robot/configs/allegro_link_info.yaml")
        )

        # Parsing chains from the urdf file
        self.chains = {}
        for finger in self.hand_configs["fingers"].keys():
            self.chains[finger] = chain.Chain.from_urdf_file(
                urdf_path,
                base_elements=[
                    self.finger_configs["links_info"]["base"]["link"],
                    self.finger_configs["links_info"][finger]["link"],
                ],
                name=finger,
            )

    def finger_forward_kinematics(self, finger_type, input_angles):
        # Checking if the number of angles is equal to 4
        if len(input_angles) != self.hand_configs["joints_per_finger"]:
            print("Incorrect number of angles")
            return

        # Checking if the input finger type is a valid one
        if finger_type not in self.hand_configs["fingers"].keys():
            print("Finger type does not exist")
            return

        # Clipping the input angles based on the finger type
        finger_info = self.finger_configs["links_info"][finger_type]
        for iterator in range(len(input_angles)):
            if input_angles[iterator] > finger_info["joint_max"][iterator]:
                input_angles[iterator] = finger_info["joint_max"][iterator]
            elif input_angles[iterator] < finger_info["joint_min"][iterator]:
                input_angles[iterator] = finger_info["joint_min"][iterator]

        # Padding values at the beginning and the end to get for a (1x6) array
        input_angles = list(input_angles)
        input_angles.insert(0, 0)
        input_angles.append(0)

        # Performing Forward Kinematics
        output_frame = self.chains[finger_type].forward_kinematics(input_angles)
        return output_frame[:3, 3], output_frame[:3, :3]

    def finger_inverse_kinematics(self, finger_type, input_position, seed=None):
        # Checking if the input figner type is a valid one
        if finger_type not in self.hand_configs["fingers"].keys():
            print("Finger type does not exist")
            return

        if seed is not None:
            # Checking if the number of angles is equal to 4
            if len(seed) != self.hand_configs["joints_per_finger"]:
                print("Incorrect seed array length")
                return

            # Clipping the input angles based on the finger type
            finger_info = self.finger_configs["links_info"][finger_type]
            for iterator in range(len(seed)):
                if seed[iterator] > finger_info["joint_max"][iterator]:
                    seed[iterator] = finger_info["joint_max"][iterator]
                elif seed[iterator] < finger_info["joint_min"][iterator]:
                    seed[iterator] = finger_info["joint_min"][iterator]

            # Padding values at the beginning and the end to get for a (1x6) array
            seed = list(seed)
            seed.insert(0, 0)
            seed.append(0)

        output_angles = self.chains[finger_type].inverse_kinematics(
            input_position, initial_position=seed
        )
        return output_angles[1:5]

    def get_fingertip_coords(self, joint_positions):
        index_coords = self.finger_forward_kinematics('index', joint_positions[:4])[0]
        middle_coords = self.finger_forward_kinematics('middle', joint_positions[4:8])[0]
        ring_coords = self.finger_forward_kinematics('ring', joint_positions[8:12])[0]
        thumb_coords = self.finger_forward_kinematics('thumb', joint_positions[12:16])[0]

        finger_tip_coords = np.hstack([index_coords, middle_coords, ring_coords, thumb_coords])
        return np.array(finger_tip_coords)
