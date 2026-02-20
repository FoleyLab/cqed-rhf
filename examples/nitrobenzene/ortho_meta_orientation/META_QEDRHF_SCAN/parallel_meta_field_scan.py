import numpy as np
import psi4
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from cqed_rhf.calculator import CQEDRHFCalculator


def generate_field_vector_from_theta_and_phi(theta, phi):
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)
    return np.array([x, y, z])


# ----------------------------
# Worker function
# ----------------------------
def compute_energy_worker(args):
    """
    Each worker process initializes its own psi4 environment and calculator.
    This avoids all shared-state issues with psi4's global configuration.
    """
    theta, phi, ortho_string, psi4_options, omega = args

    # Give each process its own psi4 output file based on PID
    pid = os.getpid()
    psi4.set_memory("4 GB")  # divide your total memory budget by n_workers
    psi4.core.set_output_file(f"psi4_worker_{pid}.out", False)
    psi4.set_options(psi4_options)

    field_vector = generate_field_vector_from_theta_and_phi(theta, phi) * 0.1

    calculator = CQEDRHFCalculator(
        lambda_vector=field_vector,
        psi4_options=psi4_options,
        omega=omega,
        density_fitting=True,
        charge=0,
        multiplicity=1
    )

    e_cqed = calculator.energy(ortho_string)
    return theta, phi, field_vector, e_cqed


def main():
    # ----------------------------
    # Configuration
    # ----------------------------
    ortho_string = """
1 1
 C                  0.02949981    1.33972592    0.06817723
 C                  1.43483278    1.28667967    0.00635313
 C                  2.11179024    0.05106117   -0.00544138
 C                  1.44506636   -1.13720058    0.03116583
 C                 -0.68793171    0.16822220    0.10995314
 H                 -0.47126997    2.29839666    0.07811355
 H                  2.02732783    2.19651728   -0.03220624
 H                  1.98966526   -2.07643217    0.02318494
 H                 -1.77163480    0.18040547    0.15819632
 N                  3.58635895    0.05097292   -0.06745286
 O                  4.14711759   -1.05966097   -0.08807849
 O                  4.14497859    1.16390951   -0.09010823
 C                 -0.02361177   -1.14582791    0.08353483
 H                 -0.43674996   -1.87247364    0.78889576
 Br                -0.53591638   -1.86972195   -1.74078671
units angstrom
no_reorient
no_com
symmetry c1
"""

    psi4_options = {
        "basis": "6-311G*",
        "scf_type": "df",
        "e_convergence": 1e-12,
        "d_convergence": 1e-12,
    }
    omega = 0.06615

    # ----------------------------
    # Core allocation
    # Adjust fraction to taste; leave headroom for the OS and main process
    # ----------------------------
    total_cores = multiprocessing.cpu_count()
    n_workers = max(1, int(total_cores * 0.3))
    # Each psi4 worker gets an equal share of total memory
    # (set "X GB" in worker to: total_ram_gb * fraction / n_workers)
    print(f"Launching {n_workers} workers across {total_cores} available cores")

    # ----------------------------
    # Special cases: theta=0 and theta=180
    # Only need one phi value each; run in the main process
    # ----------------------------
    psi4.set_memory("4 GB")
    psi4.core.set_output_file("psi4_main.out", False)
    psi4.set_options(psi4_options)

    field_vector_theta_0   = generate_field_vector_from_theta_and_phi(0, 0)   * 0.1
    field_vector_theta_180 = generate_field_vector_from_theta_and_phi(180, 0) * 0.1

    calc_main = CQEDRHFCalculator(
        lambda_vector=field_vector_theta_0,
        psi4_options=psi4_options,
        omega=omega,
        density_fitting=True,
        charge=0,
        multiplicity=1
    )
    e_cqed_theta_0 = calc_main.energy(ortho_string)
    calc_main.lambda_vector = field_vector_theta_180
    e_cqed_theta_180 = calc_main.energy(ortho_string)
    print(f"theta=0   energy: {e_cqed_theta_0:.12f}")
    print(f"theta=180 energy: {e_cqed_theta_180:.12f}")

    # ----------------------------
    # Build task list (skip redundant phi at theta=0/180)
    # ----------------------------
    theta_list = np.arange(0, 181, 8)
    phi_list   = np.arange(0, 361, 16)

    tasks = [
        (theta, phi, ortho_string, psi4_options, omega)
        for theta in theta_list
        for phi in phi_list
        if theta not in (0, 180)
    ]
    print(f"Total tasks to parallelize: {len(tasks)}")

    # ----------------------------
    # Parallel execution
    # ----------------------------
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(compute_energy_worker, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            try:
                theta, phi, field_vector, e_cqed = future.result()
                results[(theta, phi)] = (field_vector, e_cqed)
                completed += 1
                if completed % 50 == 0 or completed == len(tasks):
                    print(f"  Progress: {completed}/{len(tasks)} completed")
            except Exception as exc:
                task = futures[future]
                print(f"  ERROR for theta={task[0]}, phi={task[1]}: {exc}")

    # Populate special-case results for all phi at theta=0 and theta=180
    for phi in phi_list:
        results[(0,   phi)] = (field_vector_theta_0,   e_cqed_theta_0)
        results[(180, phi)] = (field_vector_theta_180, e_cqed_theta_180)

    # ----------------------------
    # Write output in sorted theta/phi order
    # ----------------------------
    with open("meta_field_scan_results.txt", "w") as f:
        for theta in theta_list:
            for phi in phi_list:
                fv, e_cqed = results[(theta, phi)]
                line = (f"{theta:3f} {phi:3f} "
                        f"{fv[0]: .4f} {fv[1]: .4f} {fv[2]: .4f} "
                        f"{e_cqed:.12f}")
                print(line)
                f.write(line + "\n")

    print("Done. Results written to meta_field_scan_results.txt")


if __name__ == "__main__":
    main()
