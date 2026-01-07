GEN_NAME = my_torch_generator
GEN_SRC  = src/generator.py

AN_NAME  = my_torch_analyzer
AN_SRC   = src/analyzer.py

DIST_DIR = dist
BUILD_DIR = build

all: generator analyzer

generator:
	pyinstaller --onefile --name $(GEN_NAME) --paths . --collect-submodules my_torch $(GEN_SRC)
	@cp $(DIST_DIR)/$(GEN_NAME) .

analyzer:
	pyinstaller --onefile --name $(AN_NAME) --paths . --collect-submodules my_torch $(AN_SRC)
	@cp $(DIST_DIR)/$(AN_NAME) .

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR) *.spec $(GEN_NAME) $(AN_NAME)

fclean: clean
	@true

re: fclean all