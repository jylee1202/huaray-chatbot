import os

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "huaray-manual-index")

answer_examples = [
    {
        "input": "GigE 카메라에서 제공하는 사용자 레벨은 무엇인가요?",
        "answer": "매뉴얼의 기능 파라미터 장에 따르면 카메라는 Beginner, Expert, Guru의 세 가지 사용자 레벨을 제공하며 각 레벨은 접근 가능한 설정 범위가 다릅니다."
    },
    {
        "input": "프레임 레이트는 어떤 요소의 영향을 받나요?",
        "answer": "매뉴얼 3.1절은 프레임 레이트가 대역폭, 픽셀 포맷, 이미지 해상도, 노출 시간에 의해 좌우된다고 설명합니다. 필요한 프레임 속도를 얻으려면 이 네 가지 요소를 함께 조정하세요."
    },
    {
        "input": "MultiFrame 모드에서 캡처 프레임 수는 어디에서 설정하나요?",
        "answer": "매뉴얼에 따르면 MultiFrame 모드에서는 AcquisitionFrameCount 값을 지정해야 하며, 설정한 값만큼 캡처한 뒤 수동으로 중지할 수 있습니다."
    }
]
