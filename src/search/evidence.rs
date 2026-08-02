use std::collections::{HashMap, HashSet};

pub(super) fn has_case_study_intent(query: &str) -> bool {
    let lower = query.to_ascii_lowercase();
    lower.contains("case study") || lower.contains("case studies")
}

pub(super) fn has_numerical_parity_intent(query: &str) -> bool {
    let terms = normalized_terms(query);
    terms.contains("cpu")
        && terms.contains("cuda")
        && (terms.contains("drift") || terms.contains("reproducibility"))
        && (terms.contains("numerical") || terms.contains("tolerance"))
}

/// Whether the request calls for research-grade source material rather than
/// merely using the word "study" as part of an operational case-study query.
pub(super) fn has_research_source_intent(query: &str) -> bool {
    let terms = normalized_terms(query);
    terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "evidence"
                | "paper"
                | "papers"
                | "preprint"
                | "preprints"
                | "reproducibility"
                | "research"
        )
    }) || (!has_case_study_intent(query) && (terms.contains("study") || terms.contains("studies")))
}

#[derive(Debug)]
pub(super) struct QueryEvidence {
    weights: HashMap<String, f32>,
    total_weight: f32,
    decisive_terms: Vec<String>,
    exact_identifiers: Vec<Vec<String>>,
    required_acronyms: Vec<String>,
    recommendation_intent: bool,
    full_text_intent: bool,
    as_of_intent: bool,
    timeline_intent: bool,
    requested_year: Option<i32>,
}

impl QueryEvidence {
    pub(super) fn new<'a>(query: &str, documents: impl IntoIterator<Item = &'a str>) -> Self {
        let required_acronyms = query
            .split(|character: char| !character.is_ascii_alphabetic())
            .filter(|token| {
                (3..=8).contains(&token.len())
                    && token
                        .chars()
                        .all(|character| character.is_ascii_uppercase())
            })
            .map(str::to_ascii_lowercase)
            .collect::<Vec<_>>();
        let query_words = query_tokens(query).collect::<Vec<_>>();
        let terms = lexical_terms(query);
        let document_terms = documents
            .into_iter()
            .map(normalized_terms)
            .collect::<Vec<_>>();
        let document_count = document_terms.len() as f32;
        let mut weighted_terms = terms
            .into_iter()
            .enumerate()
            .map(|(position, term)| {
                let frequency = document_terms
                    .iter()
                    .filter(|document| document.contains(&term))
                    .count();
                let weight = ((document_count + 1.0) / (frequency as f32 + 1.0)).ln() + 1.0;
                (term, weight, frequency, position)
            })
            .collect::<Vec<_>>();
        let weights = weighted_terms
            .iter()
            .map(|(term, weight, _, _)| (term.clone(), *weight))
            .collect::<HashMap<_, _>>();
        let total_weight = weights.values().sum();
        weighted_terms.retain(|(term, _, frequency, _)| {
            *frequency > 0
                && !term.chars().all(|character| character.is_ascii_digit())
                && !is_month(term)
                && !is_year(term)
                && !required_acronyms.contains(term)
        });
        weighted_terms.sort_by(
            |(_, left_weight, _, left_position), (_, right_weight, _, right_position)| {
                right_weight
                    .total_cmp(left_weight)
                    .then_with(|| left_position.cmp(right_position))
            },
        );
        let decisive_terms = weighted_terms
            .into_iter()
            .take(2)
            .map(|(term, _, _, _)| term)
            .collect();

        Self {
            weights,
            total_weight,
            decisive_terms,
            exact_identifiers: query_identifiers(query),
            required_acronyms,
            recommendation_intent: query_tokens(query).any(|token| {
                matches!(
                    token.as_str(),
                    "guideline" | "guidelines" | "recommendation" | "recommendations" | "standards"
                )
            }),
            full_text_intent: has_full_text_intent(query),
            as_of_intent: is_as_of_query(query),
            timeline_intent: query_words.iter().any(|token| {
                matches!(
                    token.as_str(),
                    "incident" | "incidents" | "lts" | "release" | "releases" | "schedule"
                )
            }),
            requested_year: years(query).next(),
        }
    }

    pub(super) fn score(&self, text: &str) -> f32 {
        let terms = normalized_terms(text);
        let lexical_score = if self.total_weight == 0.0 {
            0.0
        } else {
            let weighted_coverage = self
                .weights
                .iter()
                .filter(|(term, _)| terms.contains(*term))
                .map(|(_, weight)| weight)
                .sum::<f32>()
                / self.total_weight;
            if self.decisive_terms.is_empty() {
                weighted_coverage
            } else {
                let decisive_coverage = self
                    .decisive_terms
                    .iter()
                    .filter(|term| terms.contains(*term))
                    .count() as f32
                    / self.decisive_terms.len() as f32;
                0.60 * weighted_coverage + 0.40 * decisive_coverage
            }
        };

        if self.exact_identifiers.is_empty() {
            lexical_score
        } else {
            let identifier_coverage = self
                .exact_identifiers
                .iter()
                .filter(|identifier| contains_identifier(text, identifier))
                .count() as f32
                / self.exact_identifiers.len() as f32;
            0.65 * lexical_score + 0.35 * identifier_coverage
        }
    }

    pub(super) fn adds_terms(&self, excerpt: &str, content: &str) -> bool {
        let excerpt_terms = normalized_terms(excerpt);
        let content_terms = normalized_terms(content);
        self.weights
            .keys()
            .any(|term| excerpt_terms.contains(term) && !content_terms.contains(term))
    }

    /// Whether `text` contains one of the compound identifiers requested by
    /// the query. This is stronger evidence than independently matching the
    /// identifier's component words, and lets passage selection keep the
    /// surrounding definition or algorithm together.
    pub(super) fn has_exact_identifier(&self, text: &str) -> bool {
        self.exact_identifiers
            .iter()
            .any(|identifier| contains_identifier(text, identifier))
    }

    pub(super) fn has_recommendation_intent(&self) -> bool {
        self.recommendation_intent
    }

    pub(super) fn has_full_text_intent(&self) -> bool {
        self.full_text_intent
    }

    /// Scores whether a passage states actionable guidance rather than merely
    /// naming a guideline. The strongest query-specific term and an action
    /// predicate must occur in the same sentence-like segment.
    pub(super) fn recommendation_score(&self, text: &str) -> f32 {
        if !self.recommendation_intent || self.decisive_terms.is_empty() {
            return 0.0;
        }

        let context_terms = normalized_terms(text);
        if self.required_acronyms.iter().any(|acronym| {
            !context_terms.contains(acronym)
                && recommendation_authority(acronym).is_none_or(|(_, expansion)| {
                    !expansion.iter().all(|term| context_terms.contains(*term))
                })
        }) || self
            .requested_year
            .is_some_and(|year| !years(text).any(|candidate| candidate == year))
        {
            return 0.0;
        }

        text.split(['.', '!', '?', '\n', ';'])
            .map(normalized_terms)
            .filter(|terms| terms.iter().any(|term| is_recommendation_predicate(term)))
            .map(|terms| f32::from(terms.contains(&self.decisive_terms[0])))
            .fold(0.0, f32::max)
    }

    /// Explicit-year queries should prefer evidence that identifies that year
    /// and demote material whose only visible dates are substantially older.
    pub(super) fn year_adjustment(&self, text: &str) -> f32 {
        if self.as_of_intent && !self.timeline_intent {
            return 0.0;
        }
        let Some(requested) = self.requested_year else {
            return 0.0;
        };
        let years = years(text).collect::<HashSet<_>>();
        if years.contains(&requested) {
            0.03
        } else if years.iter().any(|year| *year <= requested - 2) {
            -0.08
        } else {
            0.0
        }
    }
}

pub(super) fn recommendation_authority(
    token: &str,
) -> Option<(&'static str, &'static [&'static str])> {
    match token.to_ascii_lowercase().as_str() {
        "aan" => Some(("AAN", &["american", "academy", "neurology"])),
        "acc" => Some(("ACC", &["american", "college", "cardiology"])),
        "acog" => Some((
            "ACOG",
            &["american", "college", "obstetricians", "gynecologists"],
        )),
        "ada" => Some(("ADA", &["american", "diabetes", "association"])),
        "aha" => Some(("AHA", &["american", "heart", "association"])),
        "cdc" => Some(("CDC", &["centers", "disease", "control", "prevention"])),
        "esc" => Some(("ESC", &["european", "society", "cardiology"])),
        "kdigo" => Some((
            "KDIGO",
            &["kidney", "disease", "improving", "global", "outcomes"],
        )),
        "nice" => Some((
            "NICE",
            &["national", "institute", "health", "care", "excellence"],
        )),
        "nih" => Some(("NIH", &["national", "institutes", "health"])),
        "uspstf" => Some((
            "USPSTF",
            // Match both "U.S. Preventive Services Task Force" and the
            // expanded "United States" spelling. The remaining words are
            // already distinctive enough to identify the issuing body.
            &["preventive", "services", "task", "force"],
        )),
        "who" => Some(("WHO", &["world", "health", "organization"])),
        _ => None,
    }
}

pub(super) fn normalized_terms(text: &str) -> HashSet<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

/// Preserve compound API names as a single piece of evidence. A bag of words
/// cannot distinguish documentation for `AbortSignal.any()` from a paragraph
/// that happens to mention an AbortSignal and later uses the ordinary word
/// "any". We accept Rust/Python-style separators and deliberately ignore
/// domain names so a requested source does not distort passage selection.
fn query_identifiers(query: &str) -> Vec<Vec<String>> {
    let mut seen = HashSet::new();
    query
        .split_whitespace()
        .filter(|token| !token.contains("://"))
        .map(|token| {
            token.trim_matches(|character: char| {
                !character.is_alphanumeric() && !matches!(character, '.' | ':' | '_')
            })
        })
        .filter(|token| token.contains(['.', ':', '_']))
        .map(|token| {
            token
                .split(['.', ':', '_'])
                .filter(|part| !part.is_empty())
                .map(str::to_ascii_lowercase)
                .collect::<Vec<_>>()
        })
        .filter(|parts| parts.len() >= 2)
        // Dotted versions, dates, and IP addresses are not identifiers and
        // must not turn ordinary passage selection into a local-only mode.
        .filter(|parts| {
            parts
                .iter()
                .any(|part| part.chars().any(char::is_alphabetic))
        })
        .filter(|parts| {
            !matches!(
                parts.last().map(String::as_str),
                Some("com" | "dev" | "io" | "net" | "org")
            )
        })
        .filter(|parts| seen.insert(parts.clone()))
        .collect()
}

fn contains_identifier(text: &str, identifier: &[String]) -> bool {
    let terms = query_tokens(text).collect::<Vec<_>>();
    terms
        .iter()
        .enumerate()
        .filter(|(_, term)| *term == &identifier[0])
        .any(|(start, _)| {
            let mut position = start + 1;
            identifier[1..].iter().all(|part| {
                // Markdown link targets can repeat an identifier component
                // between its visible pieces (`[AbortSignal](#abortsignal)`),
                // but an unrelated word means this is ordinary prose rather
                // than the compound name.
                while terms.get(position).is_some_and(|term| term != part) {
                    if !identifier.contains(&terms[position]) {
                        return false;
                    }
                    position += 1;
                }
                if terms.get(position) != Some(part) {
                    return false;
                }
                position += 1;
                true
            })
        })
}

fn lexical_terms(query: &str) -> Vec<String> {
    let terms = query_tokens(query).collect::<Vec<_>>();
    let mut seen = HashSet::new();
    terms
        .iter()
        .enumerate()
        .filter(|(position, term)| !is_explicit_as_of_date(&terms, *position, term))
        .map(|(_, term)| term.clone())
        .filter(|term| term.len() >= 3)
        .filter(|term| {
            !matches!(
                term.as_str(),
                "and"
                    | "any"
                    | "current"
                    | "data"
                    | "documentation"
                    | "for"
                    | "from"
                    | "guideline"
                    | "guidelines"
                    | "latest"
                    | "newest"
                    | "official"
                    | "original"
                    | "paper"
                    | "recent"
                    | "recently"
                    | "recommendation"
                    | "recommendations"
                    | "standards"
                    | "the"
                    | "today"
                    | "with"
            )
        })
        .filter(|term| seen.insert(term.clone()))
        .collect()
}

/// Whether a query asks for the contents of a document, rather than about a
/// full-text search feature. Splitting on punctuation intentionally makes
/// `full text` and `full-text` equivalent.
pub(super) fn has_full_text_intent(query: &str) -> bool {
    let terms = query_tokens(query).collect::<Vec<_>>();
    terms.iter().enumerate().any(|(position, term)| {
        let phrase_end = if term == "fulltext" {
            Some(position)
        } else if term == "full" && terms.get(position + 1).is_some_and(|term| term == "text") {
            Some(position + 1)
        } else {
            None
        };

        phrase_end.is_some_and(|end| {
            !terms.get(end + 1).is_some_and(|term| {
                matches!(
                    term.as_str(),
                    "index"
                        | "indexes"
                        | "indexing"
                        | "search"
                        | "searchable"
                        | "searches"
                        | "searching"
                )
            })
        })
    })
}

/// Whether freshness is part of the request. Keep this shared definition in
/// sync across candidate ranking and passage evidence so `current` and an
/// explicit `as of` clause cannot receive conflicting treatment.
pub(super) fn is_as_of_query(query: &str) -> bool {
    let terms = query_tokens(query).collect::<Vec<_>>();
    terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "current" | "latest" | "newest" | "recent" | "recently" | "today"
        )
    }) || terms
        .windows(2)
        .any(|terms| terms[0] == "as" && terms[1] == "of")
}

fn is_explicit_as_of_date(terms: &[String], position: usize, term: &str) -> bool {
    if !is_month(term) && !is_year(term) {
        return false;
    }

    terms[..position]
        .windows(2)
        .enumerate()
        .any(|(start, pair)| {
            pair[0] == "as"
                && pair[1] == "of"
                && terms[start + 2..=position]
                    .iter()
                    .all(|term| term == "the" || is_date_component(term))
        })
}

fn is_date_component(term: &str) -> bool {
    is_month(term)
        || is_year(term)
        || term.chars().all(|character| character.is_ascii_digit())
        || ["st", "nd", "rd", "th"].iter().any(|suffix| {
            term.strip_suffix(suffix).is_some_and(|number| {
                !number.is_empty() && number.chars().all(|character| character.is_ascii_digit())
            })
        })
}

fn is_year(token: &str) -> bool {
    token.len() == 4
        && token
            .parse::<i32>()
            .is_ok_and(|year| (1900..=2200).contains(&year))
}

fn is_month(token: &str) -> bool {
    matches!(
        token,
        "jan"
            | "january"
            | "feb"
            | "february"
            | "mar"
            | "march"
            | "apr"
            | "april"
            | "may"
            | "jun"
            | "june"
            | "jul"
            | "july"
            | "aug"
            | "august"
            | "sep"
            | "sept"
            | "september"
            | "oct"
            | "october"
            | "nov"
            | "november"
            | "dec"
            | "december"
    )
}

fn query_tokens(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_ascii_lowercase)
}

fn years(text: &str) -> impl Iterator<Item = i32> + '_ {
    text.split(|character: char| !character.is_ascii_digit())
        .filter_map(|token| token.parse::<i32>().ok())
        .filter(|year| (1900..=2200).contains(year))
}

fn is_recommendation_predicate(term: &str) -> bool {
    matches!(
        term,
        "advised"
            | "avoid"
            | "avoided"
            | "consider"
            | "considered"
            | "goal"
            | "indicated"
            | "initiate"
            | "initiated"
            | "must"
            | "offer"
            | "offered"
            | "prescribe"
            | "prescribed"
            | "reasonable"
            | "receive"
            | "recommended"
            | "recommends"
            | "should"
            | "start"
            | "started"
            | "target"
            | "use"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn research_source_intent_excludes_operational_case_studies() {
        for query in [
            "cross-backend numerical reproducibility testing",
            "research papers about deterministic inference",
            "controlled study of compiler optimizations",
        ] {
            assert!(has_research_source_intent(query), "missed {query:?}");
        }

        assert!(!has_research_source_intent(
            "Bazel Nix Pants tradeoffs case studies"
        ));
        assert!(!has_research_source_intent("CUDA kernel validation guide"));
    }

    #[test]
    fn rare_decisive_terms_outweigh_generic_topic_coverage() {
        let documents = [
            "2026 ADA diabetes",
            "2026 ADA diabetes",
            "2026 ADA diabetes",
            "statin therapy for adults",
        ];
        let evidence = QueryEvidence::new(
            "2026 ADA diabetes standards statin recommendations adults official",
            documents,
        );

        let generic = evidence.score("2026 ADA diabetes");
        let answer = evidence.score("statin therapy for adults");

        assert!(answer > generic, "answer={answer}, generic={generic}");
        assert!((0.0..=1.0).contains(&generic));
        assert!((0.0..=1.0).contains(&answer));
    }

    #[test]
    fn compound_identifiers_select_the_exact_api_passage() {
        let documents = [
            "The DOM Standard defines the AbortSignal interface and algorithms for events.",
            "AbortSignal.any(signals) returns a signal that follows its source signals.",
        ];
        let evidence =
            QueryEvidence::new("WHATWG DOM Standard AbortSignal.any() algorithm", documents);

        let generic = evidence.score(
            "The DOM Standard defines the AbortSignal interface. Any event listener can use it.",
        );
        let exact = evidence
            .score("The static AbortSignal.any(signals) steps run the following algorithm.");

        assert!(exact > generic, "exact={exact}, generic={generic}");
        assert_eq!(evidence.exact_identifiers, [["abortsignal", "any"]]);
    }

    #[test]
    fn dotted_versions_are_not_compound_identifiers() {
        let evidence = QueryEvidence::new("Python 3.11 release changes", std::iter::empty());

        assert!(evidence.exact_identifiers.is_empty());
        assert!(!evidence.has_exact_identifier("Python 3.11 release notes"));
    }

    #[test]
    fn recommendation_evidence_requires_an_actionable_statement() {
        let documents = [
            "ADA 2026 statin recommendations for adults",
            "Adults should receive moderate-intensity statin therapy.",
            "ADA 2026 clinical standards",
            "ADA 2026 diabetes guidance",
        ];
        let evidence = QueryEvidence::new("2026 ADA statin recommendations for adults", documents);

        assert_eq!(
            evidence.recommendation_score("Statin Treatment\nLipid Management"),
            0.0
        );
        assert_eq!(
            evidence.recommendation_score("ADA 2026 statin recs — viral breakdown ..."),
            0.0
        );
        assert_eq!(
            evidence.recommendation_score(
                "ADA 2026 guidance. Adults aged 40–75 should receive moderate-intensity statin therapy."
            ),
            1.0
        );
    }

    #[test]
    fn recommendation_evidence_requires_requested_authority_and_year() {
        let documents = [
            "ADA 2026 statin recommendations for adults",
            "Adults should receive moderate-intensity statin therapy.",
            "AACE 2026 statin guidance",
        ];
        let evidence = QueryEvidence::new("2026 ADA statin recommendations for adults", documents);

        assert_eq!(
            evidence
                .recommendation_score("Adults should receive moderate-intensity statin therapy."),
            0.0
        );
        assert_eq!(
            evidence.recommendation_score(
                "AACE 2026 guidance. Adults should receive moderate-intensity statin therapy."
            ),
            0.0
        );
        assert_eq!(
            evidence.recommendation_score(
                "ADA 2025 guidance. Adults should receive moderate-intensity statin therapy."
            ),
            0.0
        );
        assert_eq!(
            evidence.recommendation_score(
                "ADA 2026 guidance. Adults should receive moderate-intensity statin therapy."
            ),
            1.0
        );
        assert_eq!(
            evidence.recommendation_score(
                "American Diabetes Association 2026 guidance. Adults should receive moderate-intensity statin therapy."
            ),
            1.0
        );
    }

    #[test]
    fn explicit_year_evidence_demotes_stale_material() {
        let evidence = QueryEvidence::new("2026 treatment guidelines", ["guidelines"]);

        assert_eq!(evidence.year_adjustment("Updated for 2026"), 0.03);
        assert_eq!(evidence.year_adjustment("Published June 2025"), 0.0);
        assert_eq!(evidence.year_adjustment("The 2014 recommendations"), -0.08);
    }

    #[test]
    fn as_of_dates_do_not_treat_historical_mentions_as_publication_dates() {
        let evidence = QueryEvidence::new(
            "recent Europa Clipper mission status August 2026",
            ["Europa Clipper launched in 2024 and is currently in cruise"],
        );

        assert_eq!(
            evidence.year_adjustment("Europa Clipper launched in 2024 and remains in cruise"),
            0.0
        );
    }

    #[test]
    fn as_of_release_timelines_prefer_current_sections_over_stale_sections() {
        let evidence = QueryEvidence::new(
            "latest stable Rust release notes as of 2026-08-02",
            ["Rust release notes"],
        );

        assert_eq!(
            evidence.year_adjustment("Rust 1.97 was released in July 2026"),
            0.03
        );
        assert_eq!(
            evidence.year_adjustment("Rust 1.65 was released in 2022"),
            -0.08
        );
    }

    #[test]
    fn temporal_modifiers_do_not_outweigh_the_subject() {
        let evidence = QueryEvidence::new(
            "recent Europa Clipper mission status August 2026",
            [
                "Latest August 2026 space events",
                "Europa Clipper is currently in its inner cruise phase",
            ],
        );

        assert!(!evidence.weights.contains_key("recent"));
        assert!(evidence.weights.contains_key("august"));
        assert!(evidence.weights.contains_key("2026"));
        assert!(!evidence.decisive_terms.contains(&"august".to_owned()));
        assert!(!evidence.decisive_terms.contains(&"2026".to_owned()));
        assert!(
            evidence.score("Europa Clipper is currently in its inner cruise phase")
                > evidence.score("Latest August 2026 space events")
        );
    }

    #[test]
    fn full_text_intent_distinguishes_documents_from_search_features() {
        assert!(has_full_text_intent(
            "Geneva Convention Common Article 3 full text"
        ));
        assert!(has_full_text_intent(
            "CVE-2024-3094 original disclosure full-text"
        ));
        assert!(has_full_text_intent("retrieve the fulltext of RFC 8785"));
        assert!(!has_full_text_intent(
            "PostgreSQL full text search documentation"
        ));
        assert!(!has_full_text_intent("SQLite full-text indexing examples"));
    }

    #[test]
    fn recognizes_every_supported_freshness_spelling() {
        for query in [
            "latest Rust release",
            "newest CUDA advisory",
            "recent Europa status",
            "recently disclosed CVE",
            "CPI today",
            "current Node LTS",
            "Rust release as of 2026-08-02",
            "Rust release as-of 2026-08-02",
        ] {
            assert!(is_as_of_query(query), "missed {query:?}");
        }
        assert!(!is_as_of_query("Windows Server 2019 documentation"));
        assert!(!is_as_of_query("OpenAI news August 2026"));
    }

    #[test]
    fn lexical_dates_remain_evidence_but_not_decisive_terms() {
        for (query, expected) in [
            ("May Mobility autonomous vehicles", ["may", "mobility"]),
            ("March Madness bracket rules", ["march", "madness"]),
            ("Windows Server 2019 lifecycle", ["windows", "2019"]),
            (
                "latest NVIDIA Blackwell advisory August 2026",
                ["august", "2026"],
            ),
            ("current CPI inflation July 2026", ["july", "2026"]),
        ] {
            let evidence = QueryEvidence::new(query, [query]);
            for term in expected {
                assert!(
                    evidence.weights.contains_key(term),
                    "{term:?} was dropped from {query:?}"
                );
            }
            assert!(
                evidence
                    .decisive_terms
                    .iter()
                    .all(|term| !is_month(term) && !is_year(term)),
                "a date became decisive for {query:?}"
            );
        }
    }

    #[test]
    fn explicit_as_of_cutoff_is_not_lexical_evidence() {
        let evidence = QueryEvidence::new(
            "stable Rust release as of August 2nd 2026",
            ["Rust 1.90 was released in August 2026"],
        );

        assert!(!evidence.weights.contains_key("august"));
        assert!(!evidence.weights.contains_key("2026"));
        assert!(evidence.weights.contains_key("rust"));
        assert!(evidence.weights.contains_key("release"));

        let product_version = QueryEvidence::new(
            "Windows Server 2019 support status as of today",
            ["Windows Server 2019 remains supported"],
        );
        assert!(product_version.weights.contains_key("2019"));
    }

    #[test]
    fn uspstf_matches_both_us_and_expanded_authority_names() {
        let evidence = QueryEvidence::new(
            "2026 USPSTF screening recommendations",
            ["USPSTF screening guidance"],
        );

        assert_eq!(
            evidence.recommendation_score(
                "The U.S. Preventive Services Task Force 2026 recommends screening."
            ),
            1.0
        );
        assert_eq!(
            evidence.recommendation_score(
                "The United States Preventive Services Task Force 2026 recommends screening."
            ),
            1.0
        );
    }
}
