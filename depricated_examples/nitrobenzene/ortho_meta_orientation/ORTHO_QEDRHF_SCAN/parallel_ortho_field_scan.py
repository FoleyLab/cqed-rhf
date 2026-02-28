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
 C                  0.51932475    1.23303451   -0.03194925
 C                  1.94454413    1.26916358   -0.03672882
 C                  2.62037793    0.09283428   -0.02499003
 C                 -0.19603352    0.03013062    0.00102732
 H                 -0.02069420    2.17423764   -0.04336646
 H                  2.48281698    2.20891057   -0.03611879
 H                 -1.27770137    0.03990295    0.01166953
 N                  4.09213475    0.09594076    0.03662979
 O                  4.63930696   -1.02169275    0.14459220
 O                  4.66489883    1.19839699   -0.02327545
 C                  0.49428518   -1.16712649    0.02099746
 H                 -0.03251071   -2.11492669    0.05447935
 C                  1.96291176   -1.21653219   -0.02111314
 H                  2.44359113   -1.96306433    0.61513886
 Br                 2.17304025   -1.94912156   -1.90618750
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
    n_workers = max(1, int(total_cores * 0.4))
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
        charge=1,
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
    with open("ortho_field_scan_results.txt", "w") as f:
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
