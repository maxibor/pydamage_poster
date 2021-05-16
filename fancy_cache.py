import functools
import streamlit as st
from streamlit.report_thread import get_report_ctx
import time


def fancy_cache(func=None, ttl=600, unique_to_session=False, **cache_kwargs):
    """A fancier cache decorator which allows items to expire after a certain time
    as well as promises the cache values are unique to each session.
    Parameters
    ----------
    func : Callable
        If not None, the function to be cached.
    ttl : Optional[int]
        If not None, specifies the maximum number of seconds that this item will
        remain in the cache.
    unique_to_session : boolean
        If so, then hash values are unique to that session. Otherwise, use the default
        behavior which is to make the cache global across sessions.
    **cache_kwargs
        You can pass any other arguments which you might to @st.cache
    """
    # Support passing the params via function decorator, e.g.
    # @fancy_cache(ttl=10)
    if func is None:
        return lambda f: fancy_cache(
            func=f, ttl=ttl, unique_to_session=unique_to_session, **cache_kwargs
        )

    # This will behave like func by adds two dummy variables.
    dummy_func = st.cache(
        func=lambda ttl_token, session_token, *func_args, **func_kwargs: func(
            *func_args, **func_kwargs
        ),
        **cache_kwargs
    )

    # This will behave like func but with fancy caching.
    @functools.wraps(func)
    def fancy_cached_func(*func_args, **func_kwargs):
        # Create a token which changes every ttl seconds.
        ttl_token = None
        if ttl is not None:
            ttl_token = int(time.time() / ttl)

        # Create a token which is unique to each session.
        session_token = None
        if unique_to_session:
            ctx = get_report_ctx()
            session_token = ctx.session_id

        # Call the dummy func
        return dummy_func(ttl_token, session_token, *func_args, **func_kwargs)

    return fancy_cached_func
