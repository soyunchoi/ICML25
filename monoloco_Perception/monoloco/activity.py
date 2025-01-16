
# pylint: disable=too-many-statements

import math
import copy
from contextlib import contextmanager

import numpy as np
import torch
import matplotlib.pyplot as plt

from .network.process import laplace_sampling
from .visuals.pifpaf_show import KeypointPainter, image_canvas, get_pifpaf_outputs
from .visuals.printer import draw_orientation, social_distance_colors


def social_interactions(idx, centers, angles, dds, stds=None, social_distance=False,
                        n_samples=100, threshold_prob=0.25, threshold_dist=2, radii=(0.3, 0.5)):
    """
    return flag of alert if social distancing is violated
    """

    # A) Check whether people are close together
    xx = centers[idx][0]
    zz = centers[idx][1]
    distances = [math.sqrt((xx - centers[i][0]) ** 2 + (zz - centers[i][1]) ** 2)
                 for i, _ in enumerate(centers)]
    sorted_idxs = np.argsort(distances)
    indices = [idx_t for idx_t in sorted_idxs[1:]
               if distances[idx_t] <= threshold_dist]

    # B) Check whether people are looking inwards and whether there are no intrusions
    # Deterministic
    if n_samples < 2:
        for idx_t in indices:
            if check_f_formations(idx, idx_t, centers, angles,
                                  radii=radii,  # Binary value
                                  social_distance=social_distance):
                return True

    # Probabilistic
    else:
        # Samples distance
        dds = torch.tensor(dds).view(-1, 1)
        stds = torch.tensor(stds).view(-1, 1)
        # stds_te = get_task_error(dds)  # similar results to MonoLoco but lower true positive
        laplace_d = torch.cat((dds, stds), dim=1)
        samples_d = laplace_sampling(laplace_d, n_samples=n_samples)

        # Iterate over close people
        for idx_t in indices:
            f_forms = []
            for s_d in range(n_samples):
                new_centers = copy.deepcopy(centers)
                for el in (idx, idx_t):
                    delta_d = dds[el] - float(samples_d[s_d, el])
                    theta = math.atan2(new_centers[el][1], new_centers[el][0])
                    delta_x = delta_d * math.cos(theta)
                    delta_z = delta_d * math.sin(theta)
                    new_centers[el][0] += delta_x
                    new_centers[el][1] += delta_z
                f_forms.append(check_f_formations(idx, idx_t, new_centers, angles,
                                                  radii=radii,
                                                  social_distance=social_distance))
            if (sum(f_forms) / n_samples) >= threshold_prob:
                return True
    return False


def is_raising_hand(kp):
    """
    Returns flag of alert if someone raises their hand
    """
    x=0
    y=1

    nose = 0
    l_ear = 3
    l_shoulder = 5
    l_elbow = 7
    l_hand = 9
    r_ear = 4
    r_shoulder = 6
    r_elbow = 8
    r_hand = 10

    head_width = kp[x][l_ear]- kp[x][r_ear]
    head_top = (kp[y][nose] - head_width)

    l_forearm = [kp[x][l_hand] - kp[x][l_elbow], kp[y][l_hand] - kp[y][l_elbow]]
    l_arm = [kp[x][l_shoulder] - kp[x][l_elbow], kp[y][l_shoulder] - kp[y][l_elbow]]

    r_forearm = [kp[x][r_hand] - kp[x][r_elbow], kp[y][r_hand] - kp[y][r_elbow]]
    r_arm = [kp[x][r_shoulder] - kp[x][r_elbow], kp[y][r_shoulder] - kp[y][r_elbow]]

    l_angle = (90/np.pi) * np.arccos(np.dot(l_forearm/np.linalg.norm(l_forearm), l_arm/np.linalg.norm(l_arm)))
    r_angle = (90/np.pi) * np.arccos(np.dot(r_forearm/np.linalg.norm(r_forearm), r_arm/np.linalg.norm(r_arm)))

    is_l_up = kp[y][l_hand] < kp[y][l_shoulder]
    is_r_up = kp[y][r_hand] < kp[y][r_shoulder]

    l_too_close = kp[x][l_hand] <= kp[x][l_shoulder] and kp[y][l_hand]>=head_top
    r_too_close = kp[x][r_hand] >= kp[x][r_shoulder] and kp[y][r_hand]>=head_top

    is_left_risen = is_l_up and l_angle >= 30 and not l_too_close
    is_right_risen = is_r_up and r_angle >= 30 and not r_too_close

    if is_left_risen and is_right_risen:
        return 'both'

    if is_left_risen:
        return 'left'

    if is_right_risen:
        return 'right'

    return None


def check_keypoints_visibility(kp, required_points):
    """
    Check if all required keypoints are visible and valid
    """
    threshold = 0.1     # Minimum value for keypoint coordinates to be considered valid
    for point in required_points:
        # Check if both x and y coordinates are below threshold
        # which would indicate an invalid or undetected keypoint
        if abs(kp[0][point]) < threshold and abs(kp[1][point]) < threshold:
            return False
    return True


def is_walking(kp, prev_kp=None, threshold=20, min_direction_change=0.3):
    """
    Determines if a person is walking based on leg movement and direction changes
    """
    if prev_kp is None:
        return False
        
    x, y = 0, 1
    l_knee, r_knee = 13, 14
    l_ankle, r_ankle = 15, 16
    
    # Check if all lower body keypoints are visible
    required_points = [l_knee, r_knee, l_ankle, r_ankle]
    if not check_keypoints_visibility(kp, required_points):
        return False
    
    # Calculate movement direction and size
    movements = []
    directions = []
    
    for point in required_points:
        dx = kp[x][point] - prev_kp[x][point]
        dy = kp[y][point] - prev_kp[y][point]
        movement = math.sqrt(dx*dx + dy*dy)
        
        # Calculate direction only if there is movement
        if movement > 0:
            direction = math.atan2(dy, dx)  # Angle between -pi and pi
            movements.append(movement)
            directions.append(direction)
    
    if not movements:  # No movement means not walking
        return False
    
    # Check if movement size is sufficient
    avg_movement = sum(movements) / len(movements)
    if avg_movement < threshold:
        return False
    
    # Check if direction change is sufficient (opposite direction movement for left/right leg)
    if len(directions) >= 2:
        max_direction_diff = max([abs(d1 - d2) for d1 in directions for d2 in directions])
        if max_direction_diff < min_direction_change * math.pi:
            return False
    
    return True


def is_sitting(kp, threshold_ratio=1.3):
    """
    Returns True if person is sitting based on hip and knee height
    """
    
    y = 1
    hip = 11  # left hip
    knee = 13  # left knee
    ankle = 15  # left ankle
    
    # Check if all lower body keypoints are visible
    required_points = [hip, knee, ankle]
    if not check_keypoints_visibility(kp, required_points):
        return False
    
    # Determine if person is sitting based on hip and knee height difference
    hip_height = kp[y][hip]
    knee_height = kp[y][knee]
    
    return abs(hip_height - knee_height) < threshold_ratio


def is_standing(kp, threshold_vertical=1.7):
    """
    Returns True if person is in standing position
    Consider both shoulders, hips, knees, and ankles
    """
    x, y = 0, 1
    # Left/right keypoints
    l_shoulder, r_shoulder = 5, 6
    l_hip, r_hip = 11, 12
    l_knee, r_knee = 13, 14
    l_ankle, r_ankle = 15, 16

    # Check if at least one of the upper body keypoints is visible
    left_upper = check_keypoints_visibility(kp, [l_shoulder, l_hip])
    right_upper = check_keypoints_visibility(kp, [r_shoulder, r_hip])
    if not (left_upper or right_upper):
        return False
    
    # Check if both shoulders are vertical (at least one is enough)
    left_upper_vertical = left_upper and abs(kp[x][l_shoulder] - kp[x][l_hip]) < threshold_vertical
    right_upper_vertical = right_upper and abs(kp[x][r_shoulder] - kp[x][r_hip]) < threshold_vertical
    upper_vertical = left_upper_vertical or right_upper_vertical
    
    # Check if shoulders are horizontal
    if left_upper and right_upper:
        shoulder_horizontal = abs(kp[y][l_shoulder] - kp[y][r_shoulder]) < threshold_vertical * 0.6
        hip_horizontal = abs(kp[y][l_hip] - kp[y][r_hip]) < threshold_vertical * 0.6
    else:
        shoulder_horizontal = hip_horizontal = True
    
    # Check if lower body keypoints are visible
    left_lower = check_keypoints_visibility(kp, [l_knee, l_ankle])
    right_lower = check_keypoints_visibility(kp, [r_knee, r_ankle])
    has_lower_points = left_lower or right_lower
    
    if has_lower_points:
        # Check if overall vertical alignment
        left_full_vertical = left_lower and abs(kp[x][l_shoulder] - kp[x][l_ankle]) < threshold_vertical * 1.3
        right_full_vertical = right_lower and abs(kp[x][r_shoulder] - kp[x][r_ankle]) < threshold_vertical * 1.3
        full_vertical = left_full_vertical or right_full_vertical
        
        # Check if knees are straight
        left_knee_straight = left_lower and (kp[y][l_knee] > kp[y][l_ankle] * 0.95)
        right_knee_straight = right_lower and (kp[y][r_knee] > kp[y][r_ankle] * 0.95)
        knee_straight = left_knee_straight or right_knee_straight
        
        # Check if feet are grounded
        left_feet_grounded = left_lower and (kp[y][l_ankle] > 0.65 * max(kp[y]))
        right_feet_grounded = right_lower and (kp[y][r_ankle] > 0.65 * max(kp[y]))
        feet_grounded = left_feet_grounded or right_feet_grounded
        
        # Check if knees are horizontal
        if left_lower and right_lower:
            knee_horizontal = abs(kp[y][l_knee] - kp[y][r_knee]) < threshold_vertical * 0.8
        else:
            knee_horizontal = True
        
        # Overall judgment
        basic_standing = upper_vertical and (shoulder_horizontal or hip_horizontal)
        if basic_standing:
            # At least two of the lower body conditions must be met
            lower_conditions = [full_vertical, knee_straight, feet_grounded]
            return sum([1 for cond in lower_conditions if cond]) >= 2
        else:
            # If upper body is not vertical, all lower body conditions must be met
            return full_vertical and knee_straight and feet_grounded
            
    else:
        # If only upper body is visible
        return upper_vertical and (shoulder_horizontal or hip_horizontal)


def is_crouching(kp, threshold_ratio=0.5):
    """
    Returns True if person is crouching
    """
    y = 1
    hip = 11      # left hip
    knee = 13     # left knee
    ankle = 15    # left ankle
    
    # Check if hip is close to knee height and knee is bent
    hip_knee_dist = abs(kp[y][hip] - kp[y][knee])
    knee_bent = kp[y][knee] < kp[y][ankle]
    
    return hip_knee_dist < threshold_ratio and knee_bent


def check_f_formations(idx, idx_t, centers, angles, radii, social_distance=False):
    """
    Check F-formations for people close together (this function do not expect far away people):
    1) Empty space of a certain radius (no other people or themselves inside)
    2) People looking inward
    """

    # Extract centers and angles
    other_centers = np.array(
        [cent for l, cent in enumerate(centers) if l not in (idx, idx_t)])
    theta0 = angles[idx]
    theta1 = angles[idx_t]

    # Find the center of o-space as average of two candidates (based on their orientation)
    for radius in radii:
        x_0 = np.array([float(centers[idx][0]), float(centers[idx][1])])
        x_1 = np.array([float(centers[idx_t][0]), float(centers[idx_t][1])])

        mu_0 = np.array([
            float(centers[idx][0]) + radius * math.cos(theta0),
            float(centers[idx][1]) - radius * math.sin(theta0)])
        mu_1 = np.array([
            float(centers[idx_t][0]) + radius * math.cos(theta1),
            float(centers[idx_t][1]) - radius * math.sin(theta1)])
        o_c = (mu_0 + mu_1) / 2

        # 1) Verify they are looking inwards.
        # The distance between mus and the center should be less wrt the original position and the center
        d_new = np.linalg.norm(
            mu_0 - mu_1) / 2 if social_distance else np.linalg.norm(mu_0 - mu_1)
        d_0 = np.linalg.norm(x_0 - o_c)
        d_1 = np.linalg.norm(x_1 - o_c)

        # 2) Verify no intrusion for third parties
        if other_centers.size:
            other_distances = np.linalg.norm(
                other_centers - o_c.reshape(1, -1), axis=1)
        else:
            # Condition verified if no other people
            other_distances = 100 * np.ones((1, 1))

        # Binary Classification
        # if np.min(other_distances) > radius:  # Ablation without orientation
        if d_new <= min(d_0, d_1) and np.min(other_distances) > radius:
            return True
    return False


def show_activities(args, image_t, output_path, annotations, dic_out):
    """Output frontal image with poses or combined with bird eye view"""
    assert 'front' in args.output_types or 'bird' in args.output_types, "outputs allowed: front and/or bird"

    # Initialize dictionary to store activity information
    activity_info = {}

    # Collect activity information for each person
    keypoint_sets, _ = get_pifpaf_outputs(annotations)
    
    # Dictionary to store previous frame's keypoints
    if not hasattr(show_activities, 'prev_keypoints'):
        show_activities.prev_keypoints = {}

    for idx, keypoints in enumerate(keypoint_sets):
        if idx not in activity_info:
            activity_info[idx] = {}

        # Get previous frame's keypoints
        prev_kp = show_activities.prev_keypoints.get(idx)
        
        # Check each activity
        hand_status = is_raising_hand(keypoints)
        if hand_status:
            activity_info[idx]['raising_hand'] = hand_status

        if is_walking(keypoints, prev_kp):
            activity_info[idx]['walking'] = True

        if is_sitting(keypoints):
            activity_info[idx]['sitting'] = True

        if is_standing(keypoints):
            activity_info[idx]['standing'] = True

        if is_crouching(keypoints):
            activity_info[idx]['crouching'] = True

        # Check social distance
        if 'social_distance' in args.activities:
            if len(dic_out['xyz_pred']) > 1:  # Check only if there are at least two people
                social_alert = social_interactions(
                    idx, 
                    [[x[0], x[2]] for x in dic_out['xyz_pred']], 
                    dic_out['angles'],
                    dic_out['dds'],
                    stds=dic_out.get('stds_ale'),
                    social_distance=True
                )
                if social_alert:
                    activity_info[idx]['social_distance'] = True

        # Store current frame's keypoints
        show_activities.prev_keypoints[idx] = keypoints

    # Add activity_info to dic_out
    dic_out['activities'] = activity_info

    colors = ['deepskyblue' for _ in dic_out['uv_heads']]
    if 'social_distance' in args.activities:
        colors = social_distance_colors(colors, dic_out)

    angles = dic_out['angles']
    stds = dic_out['stds_ale']
    xz_centers = [[xx[0], xx[2]] for xx in dic_out['xyz_pred']]

    # Draw keypoints and orientation
    if 'front' in args.output_types:
        keypoint_sets, _ = get_pifpaf_outputs(annotations)
        uv_centers = dic_out['uv_heads']
        sizes = [abs(dic_out['uv_heads'][idx][1] - uv_s[1]) / 1.5 for idx, uv_s in
                 enumerate(dic_out['uv_shoulders'])]
        keypoint_painter = KeypointPainter(show_box=False)

        with image_canvas(image_t,
                          output_path + '.front.png',
                          show=args.show,
                          fig_width=10,
                          dpi_factor=1.0) as ax:
            keypoint_painter.keypoints(
                ax, keypoint_sets, activities=args.activities, dic_out=dic_out,
                size=image_t.size, colors=colors)
            draw_orientation(ax, uv_centers, sizes,
                             angles, colors, mode='front')

    if 'bird' in args.output_types:
        z_max = min(args.z_max, 4 + max([el[1] for el in xz_centers]))
        with bird_canvas(output_path, z_max) as ax1:
            draw_orientation(ax1, xz_centers, [], angles, colors, mode='bird')
            draw_uncertainty(ax1, xz_centers, stds)


@contextmanager
def bird_canvas(output_path, z_max):
    fig, ax = plt.subplots(1, 1)
    fig.set_tight_layout(True)
    output_path = output_path + '.bird.png'
    x_max = z_max / 1.5
    ax.plot([0, x_max], [0, z_max], 'k--')
    ax.plot([0, -x_max], [0, z_max], 'k--')
    ax.set_ylim(0, z_max + 1)
    yield ax
    fig.savefig(output_path)
    plt.close(fig)
    print('Bird-eye-view image saved')


def draw_uncertainty(ax, centers, stds):
    for idx, std in enumerate(stds):
        std = stds[idx]
        theta = math.atan2(centers[idx][1], centers[idx][0])
        delta_x = std * math.cos(theta)
        delta_z = std * math.sin(theta)
        x = (centers[idx][0] - delta_x, centers[idx][0] + delta_x)
        z = (centers[idx][1] - delta_z, centers[idx][1] + delta_z)
        ax.plot(x, z, color='g', linewidth=2.5)
