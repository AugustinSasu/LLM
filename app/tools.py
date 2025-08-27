summary_dict = None


def build_summary_dict(titles, summaries):
    global summary_dict
    summary_dict = {title: summary for title, summary in zip(titles, summaries)}


def get_summary_by_title(title: str) -> str:
    if summary_dict is None:
        return "Summary dictionary not initialized."
    return summary_dict.get(title, "Sorry, no summary available for that title.")