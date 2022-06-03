import os
import pytest

from ramp.core.initialise import initialise_inputs
from ramp.ramp_convert_old_input_files import convert_old_user_input_file


def load_usecase(j=None, fname=None):
    peak_enlarge, year_behaviour, user_list, num_profiles = initialise_inputs(
        j, fname, num_profiles=1
    )
    return user_list


class TestTransformerClass:
    def setup_method(self):
        self.input_files_to_run = [1, 2, 3]
        self.file_suffix = "_test"
        os.chdir(
            "ramp"
        )  # for legacy code to work the loading of the input file has to happen from the ramp folder
        self.py_fnames = [
            os.path.join("input_files", f"input_file_{i}.py")
            for i in self.input_files_to_run
        ]
        self.xlsx_fnames = [
            os.path.join("test", f"input_file_{i}{self.file_suffix}.xlsx")
            for i in self.input_files_to_run
        ]
        for fname in self.xlsx_fnames:
            if os.path.exists(fname):
                os.remove(fname)

    def teardown_method(self):
        auto_code_proof = (
            "# Code automatically added by ramp_convert_old_input_files.py\n"
        )
        # remove created files
        for fname in self.xlsx_fnames:
            if os.path.exists(fname):
                os.remove(fname)
        # remove additional code in legacy input files to get the appliance name from python variable names
        for fname in self.py_fnames:
            with open(fname, "r") as fp:
                lines = fp.readlines()
                if auto_code_proof in lines:
                    idx = lines.index(auto_code_proof)
            with open(fname, "w") as fp:
                fp.writelines(lines[: idx - 1])

    def test_convert_py_to_xlsx(self):
        """Convert the 3 example .py input files to xlsx and compare each appliance of each user"""
        for i, j in enumerate(self.input_files_to_run):
            old_user_list = load_usecase(j=j)
            convert_old_user_input_file(
                self.py_fnames[i], output_path="test", suffix=self.file_suffix
            )
            new_user_list = load_usecase(fname=self.xlsx_fnames[i])
            for old_user, new_user in zip(old_user_list, new_user_list):
                if old_user != new_user:
                    pytest.fail()
