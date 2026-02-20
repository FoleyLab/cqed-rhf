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
0 1
    C            0.523578368385     1.255907785448    -0.060108858355
    C            1.911089087752     1.274081125765    -0.035970404937
    C            2.586548139142     0.067568552449     0.062788051395
    C           -0.160795572442     0.047393321989     0.013540917303
    H           -0.023965515386     2.188338822606    -0.136885192421
    H            2.471267935054     2.197833002904    -0.091532234671
    H           -1.245234987349     0.039358856807    -0.005922238394
    N            4.062760228321     0.078511442088     0.089174377413
    O            4.631058688723    -0.990736191228     0.175051091672
    O            4.617724061111     1.156143880671     0.023406499990
    C            0.538237288023    -1.150954890653     0.112080262692
    H            0.001985729920    -2.091327023106     0.169370283313
    C            1.925841172126    -1.148768075020     0.137786172505
    H            2.497157976620    -2.064141160720     0.214213842495
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
    n_workers = max(1, int(total_cores * 0.8))
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
    theta_list = np.arange(0, 181, 2)
    phi_list   = np.arange(0, 361, 2)

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
    with open("nitro_field_scan_results.txt", "w") as f:
        for theta in theta_list:
            for phi in phi_list:
                fv, e_cqed = results[(theta, phi)]
                line = (f"{theta:3f} {phi:3f} "
                        f"{fv[0]: .4f} {fv[1]: .4f} {fv[2]: .4f} "
                        f"{e_cqed:.12f}")
                print(line)
                f.write(line + "\n")

    print("Done. Results written to ortho_field_scan_results.txt")


if __name__ == "__main__":
    main()
