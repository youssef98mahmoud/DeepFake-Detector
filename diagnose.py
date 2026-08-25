import os
import logging

def check_image_sizes(data_path, num_samples=20):
    from PIL import Image

    sizes = []
    checked = 0

    logging.info(f"Checking image sizes in {data_path}...")

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    sizes.append((img.size, img_path))
                    checked += 1

                    if checked >= num_samples:
                        break
                except Exception as e:
                    logging.warning(f"Could not read {file}: {e}")

        if checked >= num_samples:
            break

    if sizes:
        logging.info(f"\nSampled {len(sizes)} images:")
        for size, path in sizes:
            logging.info(f"  {size[0]}x{size[1]} - {os.path.basename(path)}")

        widths = [s[0][0] for s in sizes]
        heights = [s[0][1] for s in sizes]

        logging.info(f"\nSize statistics:")
        logging.info(f"  Width:  min={min(widths)}, max={max(widths)}, avg={sum(widths)//len(widths)}")
        logging.info(f"  Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)//len(heights)}")


def diagnose_workers(train_loader):
    import psutil
    import time

    main_pid = os.getpid()
    main_process = psutil.Process(main_pid)

    logging.info(f"Main process PID: {main_pid}")
    logging.info(f"DataLoader num_workers: {train_loader.num_workers}")

    children_before = len(main_process.children(recursive=True))
    logging.info(f"Child processes before iteration: {children_before}")

    iterator = iter(train_loader)
    time.sleep(2)

    children_after = len(main_process.children(recursive=True))
    logging.info(f"Child processes after iteration: {children_after}")

    for child in main_process.children(recursive=True):
        logging.info(f"  - Child PID {child.pid}: {child.name()}")

    if children_after > children_before:
        logging.info(f"Workers ARE spawning: {children_after - children_before} workers detected")
    else:
        logging.warning(f"Workers NOT spawning despite num_workers={train_loader.num_workers}")

    batch = next(iterator)
    logging.info(f"First batch shape: {batch[0].shape}")

