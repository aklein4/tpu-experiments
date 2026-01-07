
from handlers import HANDLERS


def main():
    
    for h in HANDLERS:
        print(h)
        print(f"Processing handler for URL: {h.url}, Subset: {h.subset}, Split: {h.split}")
        # Here you would add the logic to compile data using the handler
        # For example:
        # data = h.load_data()
        # processed_data = h.process_data(data)
        # save_data(processed_data)


if __name__ == "__main__":
    main()