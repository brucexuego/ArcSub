import {
  YOUTUBE_FALLBACK_AUDIO_FORMAT_ID,
  VIDEO_WEB_UA,
  buildBaseYtDlpArgs,
  runYtDlp,
  secondsToDurationString,
} from '../runtime.js';
import type { VideoSiteHandlerModule } from '../types.js';

function decodeJsonEscapedText(raw: string) {
  const value = String(raw || '').trim();
  if (!value) return '';
  try {
    return JSON.parse(`"${value.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`);
  } catch {
    return value;
  }
}

const handler: VideoSiteHandlerModule = {
  async parseMetadataFallback(context) {
    const pageRes = await fetch(context.url, {
      headers: {
        'User-Agent': VIDEO_WEB_UA,
        Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      },
      redirect: 'follow',
    });
    if (!pageRes.ok) {
      throw new Error(`YouTube page request failed (${pageRes.status})`);
    }

    const html = await pageRes.text();
    const watchUrl = String(pageRes.url || context.url);
    const oembedUrl = `https://www.youtube.com/oembed?url=${encodeURIComponent(watchUrl)}&format=json`;
    let oembed: any = null;
    try {
      const oembedRes = await fetch(oembedUrl, { headers: { 'User-Agent': VIDEO_WEB_UA } });
      if (oembedRes.ok) {
        oembed = await oembedRes.json();
      }
    } catch {
      oembed = null;
    }

    const title =
      String(oembed?.title || '').trim() ||
      decodeJsonEscapedText(html.match(/"title":"((?:\\.|[^"])*)"/)?.[1] || '');
    const description = decodeJsonEscapedText(html.match(/"shortDescription":"((?:\\.|[^"])*)"/)?.[1] || '');
    const uploader =
      String(oembed?.author_name || '').trim() ||
      decodeJsonEscapedText(html.match(/"ownerChannelName":"((?:\\.|[^"])*)"/)?.[1] || '');
    const viewCount = Number(html.match(/"viewCount":"(\d+)"/)?.[1] || 0);
    const lengthSeconds = Number(html.match(/"lengthSeconds":"(\d+)"/)?.[1] || 0);
    const thumbnail =
      String(oembed?.thumbnail_url || '').trim() ||
      decodeJsonEscapedText(html.match(/"thumbnailUrl":"((?:\\.|[^"])*)"/)?.[1] || '');
    const uploadDate = String(html.match(/"publishDate":"([^"]+)"/)?.[1] || '').trim().replace(/-/g, '');

    const directUrlResult = await runYtDlp([
      ...buildBaseYtDlpArgs(),
      '--impersonate', 'chrome',
      '--extractor-args', 'youtube:player_client=android_vr,android,web',
      '-f', '18',
      '-g',
      watchUrl,
    ]);
    const directUrl = directUrlResult.code === 0
      ? String(directUrlResult.stdout || '').trim().split(/\r?\n/)[0] || ''
      : '';

    return {
      title,
      description,
      uploader,
      viewCount,
      uploadDate,
      duration: lengthSeconds > 0 ? secondsToDurationString(lengthSeconds) : '',
      thumbnail,
      parseWarning:
        'YouTube restricted full format extraction for this request. ArcSub is using a fallback parser; high-resolution and audio-only options may be unavailable.',
      debug: {
        handlerStrategy: 'youtube_oembed_fallback',
        failureTriggers: context.failureTriggers,
      },
      formats: [
        {
          id: directUrl ? '18' : 'bestvideo+bestaudio/best',
          quality: directUrl ? '360p' : 'Best available',
          size: 0,
          ext: 'mp4',
          url: directUrl,
          vcodec: 'avc1.42001E',
          acodec: 'mp4a.40.2',
        },
        ...(directUrl
          ? [{
              id: 'bestvideo+bestaudio/best',
              quality: 'Best available',
              size: 0,
              ext: 'mp4',
              url: '',
              vcodec: 'unknown',
              acodec: 'unknown',
            }]
          : []),
      ],
      audioFormats: [
        {
          id: YOUTUBE_FALLBACK_AUDIO_FORMAT_ID,
          quality: 'Extract audio from fallback video',
          size: 0,
          ext: 'mp4',
          url: '',
          vcodec: 'avc1.42001E',
          acodec: 'mp4a.40.2',
        },
      ],
    };
  },
};

export default handler;
