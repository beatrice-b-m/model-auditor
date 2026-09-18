"""Exercise Jedi's editor engine against the active Python environment.

Run from the checkout for source checks, or copy this script outside the checkout
and run with a wheel environment's Python for installed-package checks.
"""

import tempfile
from pathlib import Path

import jedi


def check_editor_workflows() -> None:
    """Check signatures, argument completion, field hovers and return navigation."""
    prefix = "from model_auditor import Auditor\nauditor = Auditor()\n"
    with tempfile.TemporaryDirectory(prefix="model-auditor-jedi-") as cache:
        jedi.settings.cache_directory = cache
        # Explicit environment paths support editable installs without depending
        # on a language server's choice of project directory.
        project = jedi.Project(Path.cwd(), added_sys_path=[str(Path.cwd())])

        def script(code: str) -> jedi.Script:
            return jedi.Script(prefix + code, project=project)

        call = script("auditor.evaluate_metrics(")
        signatures = call.get_signatures()
        assert len(signatures) == 1
        assert "-> ScoreEvaluation" in signatures[0].to_string()
        assert {"inference=", "cohort=", "threshold=", "n_bootstraps="} <= {
            item.name for item in call.complete()
        }
        result = 'result = auditor.evaluate_metrics("score")\n'
        assert {"features", "to_numeric_dataframe", "plot_metric_intervals"} <= {
            item.name for item in script(result + "result.").complete()
        }
        member = script(
            result
            + 'result.features["region"].levels["N"].metrics["sensitivity"].interval'
        )
        hovers = member.help()
        assert any("pointwise bounds" in item.docstring() for item in hovers)
        definitions = member.goto(follow_imports=True)
        assert any(
            item.full_name == "model_auditor.schemas.LevelMetric.interval"
            for item in definitions
        )
        method_help = script("auditor.evaluate_calibration").help()
        assert any("left-closed" in item.docstring() for item in method_help)
        figure = script('auditor.plot_score_distributions("score")["region"][0].')
        assert {"savefig", "axes", "canvas"} <= {
            item.name for item in figure.complete()
        }
    print("Jedi signatures, completions, hovers and navigation passed.")


if __name__ == "__main__":
    check_editor_workflows()
