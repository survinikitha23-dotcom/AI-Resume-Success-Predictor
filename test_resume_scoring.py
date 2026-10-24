import unittest

from resume_scoring import calculate_keyword_match


class CalculateKeywordMatchTests(unittest.TestCase):
    def test_identical_inputs_match_every_detected_keyword(self):
        result = calculate_keyword_match(
            "Python SQL JavaScript",
            "Python SQL JavaScript",
        )

        self.assertEqual(result["percentage"], 100.0)
        self.assertEqual(result["matched_skills"], result["detected_skills"])

    def test_partial_match_uses_detected_job_keywords_as_denominator(self):
        result = calculate_keyword_match(
            "Python and SQL",
            "Python SQL JavaScript",
        )

        self.assertEqual(result["percentage"], 66.7)
        self.assertEqual(len(result["matched_skills"]), 2)
        self.assertEqual(len(result["detected_skills"]), 3)

    def test_unrelated_resume_has_no_keyword_matches(self):
        result = calculate_keyword_match(
            "Java developer with Python experience",
            "JavaScript React AWS",
        )

        self.assertEqual(result["percentage"], 0.0)
        self.assertEqual(result["matched_skills"], [])
        self.assertEqual(len(result["missing_skills"]), len(result["detected_skills"]))

    def test_different_resumes_produce_different_keyword_scores(self):
        job_description = "Python SQL Java"
        complete = calculate_keyword_match(
            "Python SQL Java",
            job_description,
        )
        partial = calculate_keyword_match("Python", job_description)

        self.assertNotEqual(complete["percentage"], partial["percentage"])
        self.assertEqual(complete["percentage"], 100.0)
        self.assertEqual(partial["percentage"], 33.3)

    def test_no_detected_keywords_returns_no_percentage(self):
        result = calculate_keyword_match(
            "A candidate seeking a role",
            "and the for a candidate developer",
        )

        self.assertEqual(result["detected_skills"], [])
        self.assertEqual(result["matched_skills"], [])
        self.assertIsNone(result["percentage"])

    def test_java_and_javascript_are_not_false_matches(self):
        java_resume_javascript_job = calculate_keyword_match(
            "Java developer",
            "JavaScript",
        )
        javascript_resume_java_job = calculate_keyword_match(
            "JavaScript developer",
            "Java",
        )

        self.assertEqual(java_resume_javascript_job["percentage"], 0.0)
        self.assertEqual(javascript_resume_java_job["percentage"], 0.0)

    def test_javascript_abbreviation_and_whitespace_are_normalized(self):
        result = calculate_keyword_match(
            "  EXPERIENCE WITH   js ",
            "JavaScript",
        )

        self.assertEqual(result["percentage"], 100.0)

    def test_matched_keywords_are_always_a_subset_of_detected_keywords(self):
        result = calculate_keyword_match(
            "Python Python SQL and Javascript",
            "Python SQL JavaScript React",
        )

        self.assertLessEqual(len(result["matched_skills"]), len(result["detected_skills"]))
        self.assertTrue(set(result["matched_skills"]).issubset(result["detected_skills"]))
        self.assertEqual(len(result["detected_skills"]), len(set(result["detected_skills"])))

    def test_job_description_extracts_expected_technical_and_soft_skills(self):
        result = calculate_keyword_match(
            "Python SQL REST API React Git machine learning communication "
            "problem solving",
            "Build services with Python, SQL, REST APIs, React, and Git. "
            "Machine learning experience, communication, and problem-solving "
            "are required.",
        )

        self.assertEqual(
            result["detected_skills"],
            [
                "Python",
                "SQL",
                "React",
                "REST APIs",
                "Git",
                "Machine Learning",
                "Communication",
                "Problem-Solving",
            ],
        )
        self.assertEqual(result["percentage"], 100.0)

    def test_generic_words_and_abbreviations_are_not_skills(self):
        result = calculate_keyword_match(
            "MI IT Title Welcome Recent Problem",
            "MI IT Title Welcome Recent Problem",
        )

        self.assertEqual(result["detected_skills"], [])
        self.assertIsNone(result["percentage"])

    def test_generic_words_do_not_inflate_a_valid_skill_score(self):
        result = calculate_keyword_match(
            "Python MI IT Title Welcome Recent Problem",
            "Python MI IT Title Welcome Recent Problem",
        )

        self.assertEqual(result["detected_skills"], ["Python"])
        self.assertEqual(result["matched_skills"], ["Python"])
        self.assertEqual(result["percentage"], 100.0)

    def test_different_job_descriptions_extract_different_skill_sets(self):
        frontend = calculate_keyword_match(
            "React JavaScript",
            "React JavaScript and REST APIs",
        )
        data = calculate_keyword_match(
            "Python SQL machine learning",
            "Python SQL and machine learning",
        )

        self.assertEqual(
            frontend["detected_skills"],
            ["JavaScript", "React", "REST APIs"],
        )
        self.assertEqual(
            data["detected_skills"],
            ["Python", "SQL", "Machine Learning"],
        )
        self.assertEqual(frontend["percentage"], 66.7)
        self.assertEqual(data["percentage"], 100.0)


if __name__ == "__main__":
    unittest.main()
