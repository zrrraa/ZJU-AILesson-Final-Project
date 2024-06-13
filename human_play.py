import gymnasium
import numpy as np
import pygame
import flappy_bird_gymnasium

"""
运行python human_play.py可以试玩一下Flappy Bird
可以通过Obs的输出验证一下各个观测值在画面中的位置
"""
def play(use_lidar=True):
    env = gymnasium.make(
        "FlappyBird-v0", audio_on=False, render_mode="human", use_lidar=use_lidar
    )

    steps = 0
    video_buffer = []

    obs = env.reset()
    while True:
        # Getting action:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN and (
                event.key == pygame.K_SPACE or event.key == pygame.K_UP
            ):
                action = 1

        # Processing:
        obs, _, done, _, info = env.step(action)
        video_buffer.append(obs)

        steps += 1
        print(
            f"Obs: {obs}\n"
            f"Action: {action}\n"
            f"Score: {info['score']}\n Steps: {steps}\n"
        )

        if done:
            break

    env.close()


if __name__ == "__main__":
    play(use_lidar=False)