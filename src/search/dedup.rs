use std::cmp::Ordering;
use std::collections::HashSet;

use super::Raw;
use super::evidence::QueryEvidence;

const SHINGLE_WORDS: usize = 5;
const MIN_SHINGLES: usize = 40;
const MIN_CONTAINMENT_PERCENT: usize = 95;
const MIN_LARGER_COVERAGE_PERCENT: usize = 82;
const CROSS_HOST_MIN_CONTAINMENT_PERCENT: usize = 98;
const CROSS_HOST_MIN_LARGER_COVERAGE_PERCENT: usize = 95;
const MIRROR_TITLE_MIN_CONTAINMENT_PERCENT: usize = 95;
const MIRROR_TITLE_MIN_LARGER_COVERAGE_PERCENT: usize = 80;
const MAX_EXACT_DOCUMENT_MANIFESTATIONS: usize = 1;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Removes extracted copies that would otherwise spend multiple result slots
/// on substantially the same page. The deliberately high threshold keeps this
/// separate from ordinary topical similarity: two pages must share almost all
/// of the smaller page and most of the larger one.
pub(super) fn refine(query: &str, mut raw: Vec<Raw>) -> Vec<Raw> {
    prefer_source_roots(query, &mut raw);
    let raw = collapse_exact_document_manifestations(query, raw);
    suppress_near_duplicates(query, raw)
}

fn prefer_source_roots(query: &str, raw: &mut [Raw]) {
    if !super::searxng::has_strict_source_intent(&query.to_ascii_lowercase()) {
        return;
    }

    let responses = raw.iter().map(is_response_record).collect::<Vec<_>>();
    let subjects = raw.iter().map(normalized_subject).collect::<Vec<_>>();
    let hosts = raw.iter().map(normalized_host).collect::<Vec<_>>();
    let mut matched_responses = vec![false; raw.len()];
    let mut matched_roots = vec![false; raw.len()];
    for response in 0..raw.len() {
        if !responses[response] || !raw[response].source_priority {
            continue;
        }
        for root in 0..raw.len() {
            if responses[root]
                || hosts[response].is_none()
                || hosts[response] != hosts[root]
                || !equivalent_subjects(&subjects[response], &subjects[root])
            {
                continue;
            }
            matched_responses[response] = true;
            matched_roots[root] = true;
        }
    }

    // A focused source lookup can return both an archive root and a reply that
    // quotes it. Transfer the verified-source reservation only to a root with
    // the same normalized subject on the exact same host. Merely sharing an
    // archive host is not evidence that two records describe the same event.
    for (index, document) in raw.iter_mut().enumerate() {
        if matched_responses[index] {
            document.source_priority = false;
        } else if matched_roots[index] {
            document.source_priority = true;
        }
    }

    // Stable sorting retains upstream order within each class. A response is
    // still kept (and remains reservable when it is the only source record),
    // but a verified root wins the first-party slot when both were found.
    raw.sort_by_key(|document| {
        let response = is_response_record(document);
        match (document.source_priority, response) {
            (true, false) => 0,
            (true, true) => 1,
            (false, false) => 2,
            (false, true) => 3,
        }
    });
}

fn collapse_exact_document_manifestations(query: &str, raw: Vec<Raw>) -> Vec<Raw> {
    let exact = raw
        .iter()
        .enumerate()
        .filter(|(_, document)| super::candidate::exact_document_match(query, &document.hit))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if exact.len() <= MAX_EXACT_DOCUMENT_MANIFESTATIONS {
        return raw;
    }

    let evidence = QueryEvidence::new(query, raw.iter().map(|document| document.content.as_str()));
    let evidence_scores = raw
        .iter()
        .map(|document| {
            let excerpt = format!("{}\n{}", document.hit.title, document.hit.snippet);
            evidence
                .score(&document.content)
                .max(evidence.score(&excerpt))
        })
        .collect::<Vec<_>>();
    let winner = exact
        .iter()
        .copied()
        .reduce(|left, right| {
            if exact_manifestation_order(
                &raw[left],
                evidence_scores[left],
                left,
                &raw[right],
                evidence_scores[right],
                right,
            ) == Ordering::Less
            {
                right
            } else {
                left
            }
        })
        .expect("multiple exact manifestations are non-empty");
    let first = *exact
        .first()
        .expect("multiple exact manifestations are non-empty");
    let source_priority = exact.iter().any(|index| raw[*index].source_priority);
    let upstream_consensus = exact.iter().any(|index| raw[*index].upstream_consensus);
    let exact = exact.into_iter().collect::<HashSet<_>>();
    let mut raw = raw.into_iter().map(Some).collect::<Vec<_>>();
    let mut winner = raw[winner]
        .take()
        .expect("exact manifestation winner is present");
    winner.source_priority |= source_priority;
    winner.upstream_consensus |= upstream_consensus;
    let mut winner = Some(winner);

    raw.into_iter()
        .enumerate()
        .filter_map(|(index, document)| {
            if index == first {
                winner.take()
            } else if exact.contains(&index) {
                None
            } else {
                document
            }
        })
        .collect()
}

fn exact_manifestation_order(
    left: &Raw,
    left_evidence: f32,
    left_index: usize,
    right: &Raw,
    right_evidence: f32,
    right_index: usize,
) -> Ordering {
    (!left.from_snippet)
        .cmp(&!right.from_snippet)
        .then_with(|| left.source_priority.cmp(&right.source_priority))
        .then_with(|| left_evidence.total_cmp(&right_evidence))
        .then_with(|| left.upstream_consensus.cmp(&right.upstream_consensus))
        .then_with(|| left.full_length.cmp(&right.full_length))
        .then_with(|| right_index.cmp(&left_index))
}

fn suppress_near_duplicates(query: &str, raw: Vec<Raw>) -> Vec<Raw> {
    if raw.len() < 2 {
        return raw;
    }

    let evidence = QueryEvidence::new(query, raw.iter().map(|document| document.content.as_str()));
    let evidence_scores = raw
        .iter()
        .map(|document| {
            let excerpt = format!("{}\n{}", document.hit.title, document.hit.snippet);
            evidence
                .score(&document.content)
                .max(evidence.score(&excerpt))
        })
        .collect::<Vec<_>>();
    let shingles = raw.iter().map(content_shingles).collect::<Vec<_>>();
    let subjects = raw.iter().map(normalized_subject).collect::<Vec<_>>();
    let exact_documents = raw
        .iter()
        .map(|document| super::candidate::exact_document_match(query, &document.hit))
        .collect::<Vec<_>>();

    let mut sets = DisjointSets::new(raw.len());
    for left in 0..raw.len() {
        for right in left + 1..raw.len() {
            // Keep the selected exact-title representative distinct from
            // merely related pages. A title parser may recognize only one
            // side, so ambiguity favors retaining the related candidate.
            if exact_documents[left] || exact_documents[right] {
                continue;
            }
            let (Some(left_host), Some(right_host)) =
                (registrable_host(&raw[left]), registrable_host(&raw[right]))
            else {
                continue;
            };
            let (Some(left_shingles), Some(right_shingles)) = (&shingles[left], &shingles[right])
            else {
                continue;
            };
            let same_host = left_host == right_host;
            let equivalent_title = equivalent_subjects(&subjects[left], &subjects[right]);
            if near_duplicate(left_shingles, right_shingles, same_host, equivalent_title) {
                sets.union(left, right);
            }
        }
    }

    let mut clusters = Vec::<(usize, Vec<usize>)>::new();
    for document in 0..raw.len() {
        let root = sets.find(document);
        if let Some((_, members)) = clusters.iter_mut().find(|(cluster, _)| *cluster == root) {
            members.push(document);
        } else {
            clusters.push((root, vec![document]));
        }
    }

    let strict_source = super::searxng::has_strict_source_intent(&query.to_ascii_lowercase());
    let retained = clusters
        .into_iter()
        .map(|(_, members)| {
            let winner = members
                .iter()
                .copied()
                .reduce(|left, right| {
                    if representative_order(
                        strict_source,
                        &raw[left],
                        evidence_scores[left],
                        left,
                        &raw[right],
                        evidence_scores[right],
                        right,
                    ) == Ordering::Less
                    {
                        right
                    } else {
                        left
                    }
                })
                .expect("duplicate cluster is non-empty");
            let original_position = *members
                .iter()
                .min()
                .expect("duplicate cluster is non-empty");
            let source_priority = members
                .iter()
                .any(|document| raw[*document].source_priority);
            let upstream_consensus = members
                .iter()
                .any(|document| raw[*document].upstream_consensus);
            (
                original_position,
                winner,
                source_priority,
                upstream_consensus,
            )
        })
        .collect::<Vec<_>>();
    let mut raw = raw.into_iter().map(Some).collect::<Vec<_>>();
    let mut retained = retained
        .into_iter()
        .map(
            |(original_position, winner, source_priority, upstream_consensus)| {
                let mut document = raw[winner]
                    .take()
                    .expect("duplicate representatives are unique");
                document.source_priority |= source_priority;
                document.upstream_consensus |= upstream_consensus;
                (original_position, document)
            },
        )
        .collect::<Vec<_>>();
    retained.sort_by_key(|(original_position, _)| *original_position);
    retained.into_iter().map(|(_, document)| document).collect()
}

fn representative_order(
    strict_source: bool,
    left: &Raw,
    left_evidence: f32,
    left_index: usize,
    right: &Raw,
    right_evidence: f32,
    right_index: usize,
) -> Ordering {
    let left_root = !strict_source || !is_response_record(left);
    let right_root = !strict_source || !is_response_record(right);
    left_root
        .cmp(&right_root)
        .then_with(|| left.source_priority.cmp(&right.source_priority))
        .then_with(|| (!left.from_snippet).cmp(&!right.from_snippet))
        .then_with(|| left_evidence.total_cmp(&right_evidence))
        .then_with(|| left.upstream_consensus.cmp(&right.upstream_consensus))
        .then_with(|| left.full_length.cmp(&right.full_length))
        // Earlier upstream position wins a complete tie.
        .then_with(|| right_index.cmp(&left_index))
}

fn content_shingles(document: &Raw) -> Option<HashSet<u64>> {
    if document.from_snippet {
        return None;
    }
    let words = document
        .content
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(normalized_word_hash)
        .collect::<Vec<_>>();
    if words.len() < MIN_SHINGLES + SHINGLE_WORDS - 1 {
        return None;
    }

    // Store hashes rather than shingle strings, but retain the complete set.
    // Exact containment thresholds are meaningful only when every shingle is
    // represented; bottom-hash sampling can make unrelated page tails vanish.
    let shingles = words
        .windows(SHINGLE_WORDS)
        .map(shingle_hash)
        .collect::<HashSet<_>>();
    (shingles.len() >= MIN_SHINGLES).then_some(shingles)
}

fn normalized_word_hash(word: &str) -> u64 {
    let mut hash = FNV_OFFSET;
    for character in word.chars().flat_map(char::to_lowercase) {
        let mut encoded = [0; 4];
        for byte in character.encode_utf8(&mut encoded).bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

fn shingle_hash(words: &[u64]) -> u64 {
    words.iter().fold(FNV_OFFSET, |mut hash, word| {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    })
}

fn near_duplicate(
    left: &HashSet<u64>,
    right: &HashSet<u64>,
    same_host: bool,
    equivalent_title: bool,
) -> bool {
    let intersection = left.intersection(right).count();
    let smaller = left.len().min(right.len());
    let larger = left.len().max(right.len());
    let (containment, larger_coverage) = if same_host {
        (MIN_CONTAINMENT_PERCENT, MIN_LARGER_COVERAGE_PERCENT)
    } else if equivalent_title {
        (
            MIRROR_TITLE_MIN_CONTAINMENT_PERCENT,
            MIRROR_TITLE_MIN_LARGER_COVERAGE_PERCENT,
        )
    } else {
        (
            CROSS_HOST_MIN_CONTAINMENT_PERCENT,
            CROSS_HOST_MIN_LARGER_COVERAGE_PERCENT,
        )
    };
    intersection * 100 >= smaller * containment && intersection * 100 >= larger * larger_coverage
}

fn registrable_host(document: &Raw) -> Option<String> {
    document
        .hit
        .url
        .host_str()
        .map(super::searxng::registrable_host)
}

fn normalized_host(document: &Raw) -> Option<String> {
    document.hit.url.host_str().map(str::to_ascii_lowercase)
}

fn normalized_subject(document: &Raw) -> Vec<String> {
    let (_, title) = response_marker_and_subject(&document.hit.title);
    title
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

fn equivalent_subjects(left: &[String], right: &[String]) -> bool {
    if left.is_empty() || right.is_empty() {
        return false;
    }
    if left == right {
        return true;
    }

    let (shorter, longer) = if left.len() < right.len() {
        (left, right)
    } else {
        (right, left)
    };
    shorter.len() >= 4
        && shorter.len() * 4 >= longer.len() * 3
        && (longer.starts_with(shorter) || longer.ends_with(shorter))
}

fn response_marker_and_subject(mut title: &str) -> (bool, &str) {
    let mut response = false;
    loop {
        let trimmed = title.trim_start();
        if let Some((_, rest)) = trimmed
            .strip_prefix('[')
            .and_then(|title| title.split_once(']'))
        {
            title = rest;
            continue;
        }
        let Some(prefix) = RESPONSE_PREFIXES.iter().find(|prefix| {
            trimmed
                .get(..prefix.len())
                .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
        }) else {
            return (response, trimmed);
        };
        response = true;
        title = &trimmed[prefix.len()..];
    }
}

const RESPONSE_PREFIXES: &[&str] = &[
    "comment on:",
    "comment on ",
    "comment:",
    "comments on ",
    "fwd:",
    "fw:",
    "re:",
    "reply to:",
    "reply to ",
    "reply:",
    "response to:",
    "response to ",
    "response:",
];

fn is_response_record(document: &Raw) -> bool {
    response_marker_and_subject(&document.hit.title).0
        || document.hit.url.query_pairs().any(|(name, _)| {
            matches!(
                name.to_ascii_lowercase().as_ref(),
                "comment" | "commentid" | "reply" | "replytocom" | "response"
            )
        })
}

struct DisjointSets {
    parents: Vec<usize>,
}

impl DisjointSets {
    fn new(len: usize) -> Self {
        Self {
            parents: (0..len).collect(),
        }
    }

    fn find(&mut self, index: usize) -> usize {
        let parent = self.parents[index];
        if parent == index {
            index
        } else {
            let root = self.find(parent);
            self.parents[index] = root;
            root
        }
    }

    fn union(&mut self, left: usize, right: usize) {
        let left = self.find(left);
        let right = self.find(right);
        if left != right {
            self.parents[right] = left;
        }
    }
}

#[cfg(test)]
mod tests {
    use url::Url;

    use super::*;
    use crate::search::searxng::Hit;

    fn raw(title: &str, url: &str, content: &str) -> Raw {
        Raw {
            hit: Hit {
                title: title.into(),
                url: Url::parse(url).unwrap(),
                date: None,
                snippet: String::new(),
            },
            source_priority: false,
            upstream_consensus: false,
            content: content.into(),
            full_length: content.chars().count(),
            from_snippet: false,
        }
    }

    fn numbered_content(start: usize, end: usize) -> String {
        (start..end)
            .map(|number| format!("distinctword{number}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn strict_source_queries_transfer_priority_from_reply_to_root() {
        let mut reply = raw(
            "[oss-security] Re: upstream security disclosure",
            "https://archive.example/list/10",
            &numbered_content(0, 100),
        );
        reply.source_priority = true;
        let root = raw(
            "upstream security disclosure",
            "https://archive.example/list/2",
            &numbered_content(200, 300),
        );

        let refined = refine(
            "original disclosure from archive.example",
            vec![reply, root],
        );

        assert_eq!(refined[0].hit.title, "upstream security disclosure");
        assert!(refined[0].source_priority);
        assert!(!refined[1].source_priority);
    }

    #[test]
    fn lone_reply_remains_a_usable_requested_source() {
        let mut reply = raw(
            "Re: upstream security disclosure",
            "https://archive.example/list/10",
            &numbered_content(0, 100),
        );
        reply.source_priority = true;

        let refined = refine("original disclosure from archive.example", vec![reply]);

        assert!(refined[0].source_priority);
    }

    #[test]
    fn unrelated_root_on_the_same_host_does_not_steal_reply_priority() {
        let mut reply = raw(
            "[oss-security] Re: upstream security disclosure",
            "https://archive.example/list/10",
            &numbered_content(0, 100),
        );
        reply.source_priority = true;
        let unrelated = raw(
            "Quarterly archive maintenance notice",
            "https://archive.example/list/2",
            &numbered_content(200, 300),
        );

        let refined = refine(
            "original disclosure from archive.example",
            vec![reply, unrelated],
        );

        let reply = refined
            .iter()
            .find(|document| document.hit.title.contains("security disclosure"))
            .unwrap();
        let unrelated = refined
            .iter()
            .find(|document| document.hit.title.contains("maintenance"))
            .unwrap();
        assert!(reply.source_priority);
        assert!(!unrelated.source_priority);
    }

    #[test]
    fn matching_subject_on_a_sibling_host_does_not_steal_reply_priority() {
        let mut reply = raw(
            "Re: upstream security disclosure",
            "https://archive.example/list/10",
            &numbered_content(0, 100),
        );
        reply.source_priority = true;
        let sibling = raw(
            "upstream security disclosure",
            "https://mirror.archive.example/list/2",
            &numbered_content(200, 300),
        );

        let refined = refine(
            "original disclosure from archive.example",
            vec![reply, sibling],
        );

        let reply = refined
            .iter()
            .find(|document| document.hit.url.host_str() == Some("archive.example"))
            .unwrap();
        let sibling = refined
            .iter()
            .find(|document| document.hit.url.host_str() == Some("mirror.archive.example"))
            .unwrap();
        assert!(reply.source_priority);
        assert!(!sibling.source_priority);
    }

    #[test]
    fn near_duplicate_reply_is_removed_without_displacing_the_root() {
        let content = numbered_content(0, 120);
        let mut reply = raw(
            "Re: upstream security disclosure",
            "https://archive.example/list/10",
            &content,
        );
        reply.source_priority = true;
        let root = raw(
            "upstream security disclosure",
            "https://archive.example/list/2",
            &content,
        );

        let refined = refine("original disclosure", vec![reply, root]);

        assert_eq!(refined.len(), 1);
        assert_eq!(refined[0].hit.title, "upstream security disclosure");
        assert!(refined[0].source_priority);
    }

    #[test]
    fn suppresses_high_overlap_copies_while_retaining_related_pages() {
        let original = numbered_content(0, 120);
        let mut changed = numbered_content(0, 116);
        changed.push_str(" replacement116 replacement117 replacement118 replacement119");
        let related = numbered_content(0, 80) + " " + &numbered_content(500, 540);
        let other_domain = changed.clone();

        let refined = refine(
            "distinctword40",
            vec![
                raw("canonical", "https://docs.example/a", &original),
                raw("copy", "https://docs.example/b", &changed),
                raw("related", "https://docs.example/c", &related),
                raw("mirror", "https://other.example/d", &other_domain),
            ],
        );

        assert_eq!(refined.len(), 2);
        assert!(
            refined
                .iter()
                .any(|document| document.hit.title == "related")
        );
    }

    #[test]
    fn suppresses_high_overlap_pages_with_more_than_4096_shingles() {
        let original = numbered_content(0, 5_000);
        let mut changed = numbered_content(0, 4_990);
        changed.push(' ');
        changed.push_str(&numbered_content(9_000, 9_010));
        let canonical = raw("canonical", "https://docs.example/a", &original);
        assert!(content_shingles(&canonical).unwrap().len() > 4_096);

        let refined = refine(
            "distinctword40",
            vec![canonical, raw("copy", "https://docs.example/b", &changed)],
        );

        assert_eq!(refined.len(), 1);
    }

    #[test]
    fn detects_large_page_containment_using_the_complete_shingle_sets() {
        let contained = numbered_content(0, 4_200);
        let containing = numbered_content(0, 4_500);
        let contained = raw("contained", "https://docs.example/a", &contained);
        assert!(content_shingles(&contained).unwrap().len() > 4_096);

        let refined = refine(
            "distinctword40",
            vec![
                contained,
                raw("containing", "https://docs.example/b", &containing),
            ],
        );

        assert_eq!(refined.len(), 1);
    }

    #[test]
    fn suppresses_near_identical_cross_host_mirrors() {
        let original = numbered_content(0, 5_000);
        let mut mirror = numbered_content(0, 4_990);
        mirror.push(' ');
        mirror.push_str(&numbered_content(9_000, 9_010));

        let refined = refine(
            "distinctword40",
            vec![
                raw("archive record", "https://archive.example/a", &original),
                raw(
                    "GitHub mirror",
                    "https://github.com/example/mirror",
                    &mirror,
                ),
            ],
        );

        assert_eq!(refined.len(), 1);
    }

    #[test]
    fn suppresses_titled_cross_host_mirror_with_archive_framing() {
        let original = numbered_content(0, 100);
        let mirror = format!("{original} {}", numbered_content(1_000, 1_020));

        let refined = refine(
            "AbortSignal.any assertion failure",
            vec![
                raw(
                    "[whatwg/dom] AbortSignal.any assertion failure (Issue #1293)",
                    "https://lists.w3.org/archive/1293",
                    &mirror,
                ),
                raw(
                    "AbortSignal.any assertion failure Issue #1293 whatwg/dom",
                    "https://github.com/whatwg/dom/issues/1293",
                    &original,
                ),
            ],
        );

        assert_eq!(refined.len(), 1);
    }

    #[test]
    fn retains_cross_host_commentary_with_a_substantial_unique_tail() {
        let shared = numbered_content(0, 4_800);
        let left = format!("{shared} {}", numbered_content(9_000, 9_500));
        let right = format!("{shared} {}", numbered_content(10_000, 10_500));

        let refined = refine(
            "distinctword40",
            vec![
                raw("publisher", "https://publisher.example/a", &left),
                raw("commentary", "https://commentary.example/b", &right),
            ],
        );

        assert_eq!(refined.len(), 2);
    }

    #[test]
    fn duplicate_cluster_keeps_stronger_evidence_and_merges_consensus() {
        let generic = numbered_content(0, 120);
        let mut relevant = generic.clone();
        relevant.push_str(" targetphrase");
        let mut weak = raw("generic copy", "https://docs.example/a", &generic);
        weak.upstream_consensus = true;
        let strong = raw(
            "targetphrase official details",
            "https://docs.example/b",
            &relevant,
        );

        let refined = refine("targetphrase details", vec![weak, strong]);

        assert_eq!(refined.len(), 1);
        assert_eq!(refined[0].hit.title, "targetphrase official details");
        assert!(refined[0].upstream_consensus);
    }

    #[test]
    fn duplicate_cluster_prefers_verified_authority_over_lexical_strength() {
        let content = numbered_content(0, 120);
        let mut authority = raw("Official record", "https://docs.example/official", &content);
        authority.source_priority = true;
        let lexical = raw(
            "targetphrase details",
            "https://docs.example/copy",
            &content,
        );

        let refined = refine("targetphrase details official", vec![authority, lexical]);

        assert_eq!(refined.len(), 1);
        assert_eq!(refined[0].hit.title, "Official record");
    }

    #[test]
    fn exact_document_manifestations_collapse_after_fetch_and_backfill_the_tail() {
        let mut inaccessible = raw(
            "Attention Is All You Need",
            "https://arxiv.org/abs/1706.03762",
            "search result snippet",
        );
        inaccessible.from_snippet = true;
        let preferred = raw(
            "Attention Is All You Need",
            "https://arxiv.org/pdf/1706.03762",
            &numbered_content(0, 120),
        );
        let mut inaccessible_mirror = raw(
            "Attention Is All You Need | NeurIPS",
            "https://proceedings.neurips.cc/paper/attention-is-all-you-need",
            "another search result snippet",
        );
        inaccessible_mirror.from_snippet = true;
        let related_one = raw(
            "A survey of transformer architectures",
            "https://related.example/survey",
            &numbered_content(200, 320),
        );
        let related_two = raw(
            "Implementing multi-head attention",
            "https://related.example/implementation",
            &numbered_content(400, 520),
        );

        let refined = refine(
            "Attention Is All You Need original paper PDF",
            vec![
                inaccessible,
                related_one,
                preferred,
                inaccessible_mirror,
                related_two,
            ],
        );

        assert_eq!(refined.len(), 3);
        assert_eq!(
            refined
                .iter()
                .filter(|document| super::super::candidate::exact_document_match(
                    "Attention Is All You Need original paper PDF",
                    &document.hit,
                ))
                .count(),
            1
        );
        assert!(
            refined
                .iter()
                .find(|document| document.hit.title == "Attention Is All You Need")
                .is_some_and(|document| !document.from_snippet)
        );
        assert!(
            refined
                .iter()
                .any(|document| document.hit.title.contains("survey"))
        );
        assert!(
            refined
                .iter()
                .any(|document| document.hit.title.contains("multi-head"))
        );
    }
}
